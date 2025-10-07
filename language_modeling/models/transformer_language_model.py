import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, List
from framework.layers import LoggingLayer, RegularizedLayer
from layers.transformer.multi_head_attention import AttentionMask
from layers.transformer.transformer import generate_square_subsequent_mask
from layers import MoE
from layers.transformer.full_moe_relative_attention import FullMoeRelativeAttentionCore
import framework
import math
import torch.nn.utils.parametrizations


def unique_obejcts(l: List[Any]):
    res = []
    known = set()
    for o in l:
        if id(o) not in known:
            res.append(o)
            known.add(id(o))
    return res


class TransformerLanguageModel(LoggingLayer, RegularizedLayer, torch.nn.Module):
    def __init__(self, voc_size: int, embedding_size: Optional[int], state_size: int, dropout: float,
                 layers: List[torch.nn.Module], n_prev_states: int,
                 n_prev_states_test: Optional[int] = None,
                 same_length_eval: bool = True, norm_before_output: bool = False,
                 p_drop_layer: float = 0.0, use_last_state: bool = False, same_length: bool = False,
                 norm_input: bool = False,
                 cross_layer_state: bool = False,
                 log_interval: Optional[int] = 100):

        super().__init__()

        self.embedding = torch.nn.Embedding(voc_size, embedding_size or state_size)
        # with torch.no_grad():
        #     self.embedding.weight.uniform_(-0.1, 0.1)

        # torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in", nonlinearity="linear")

        self.shared_layers = all([la is layers[0] for la in layers])

        if embedding_size is None:
            self.embedding_adapter = lambda x: x
            self.output_adapter = lambda x: x
        else:
            self.embedding_adapter = torch.nn.Linear(embedding_size, state_size)
            self.output_adapter = torch.nn.Linear(state_size, embedding_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.layers = layers
        self.unique_layers = torch.nn.ModuleList(unique_obejcts(layers))
        self.n_prev_states = n_prev_states
        self.n_prev_states_test = n_prev_states_test or n_prev_states
        self.same_length_eval = same_length_eval
        self.embedding_scale = math.sqrt(state_size)
        self.p_drop_layer = p_drop_layer
        self.use_last_state = use_last_state
        self.same_length = same_length
        self.iter = 0
        self.cross_layer_state = cross_layer_state
        self.log_interval = log_interval
        self.reg_loss = 0


        out_proj_size = embedding_size or state_size
        self.output = torch.nn.Linear(out_proj_size, voc_size)

        if norm_before_output:
            self.out_norm = torch.nn.LayerNorm(state_size)
        else:
            self.out_norm = lambda x: x

        if norm_input:
            self.in_norm = torch.nn.LayerNorm(state_size, elementwise_affine=False)
        else:
            self.in_norm = lambda x: x

    @staticmethod
    def generate_history_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=-1)

    def gen_output(self, x: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        net = self.out_norm(x)
        net = self.output_adapter(net)
        net = self.dropout(net)
        net = self.output(net.transpose(0, 1))

        return net

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor], state) -> Tuple[torch.Tensor, Any]:
        causality_mask = generate_square_subsequent_mask(x.shape[0], x.device)

        net = self.dropout(self.embedding(x.T.long()))
        net = self.embedding_adapter(net)
        net = net * self.embedding_scale

        net = self.in_norm(net)

        new_state = []
        features = [net]

        n_prev_states = self.n_prev_states if self.training else self.n_prev_states_test

        same_length = self.same_length or ((not self.training) and self.same_length_eval)
        if same_length and state is not None:
            causality_mask = [self.generate_history_mask(x.shape[0], x.device)] + \
                             [torch.zeros_like(causality_mask)] * (len(state[0]) - 1) + [causality_mask]
            causality_mask = torch.cat(causality_mask, -1)


        plot_cossim = self.log_interval and self.iter % self.log_interval == 0 and self.training

        cross_layer_state = None

        for li, l in enumerate(self.layers):
            for m in l.modules():
                # Hack: make sure the layer index is right in the MoE layers. Used for local layer
                # embeddings
                if isinstance(m, (MoE, FullMoeRelativeAttentionCore)):
                    m.layer = li

            if n_prev_states > 0:
                if li == 0:
                    # Pos offset should be constant for all layers
                    pos_offset = sum(s.shape[1] for s in state[0]) if state is not None else 0

                # Concatenate the new state with the previous states
                li_r = -1 if self.use_last_state else li
                s = (state[li_r] + [net]) if state is not None else [net]
                attend_to = torch.cat(s, 1)

                if not self.use_last_state:
                    s[-1] = s[-1].detach()
                    new_state.append(s[-n_prev_states:])
            else:
                pos_offset = None
                attend_to = None

            inp = net
            mask = AttentionMask(None, causality_mask)

            if self.cross_layer_state:
                cross_layer_state, net_o = l(cross_layer_state, inp, mask=mask, attend_to=attend_to,
                                             pos_offset=pos_offset)
            else:
                net_o = l(inp, mask=mask, attend_to=attend_to, pos_offset=pos_offset)

            if plot_cossim:
                features.append(net_o)

            with torch.no_grad():
                ndiff = torch.norm(net_o - net, p=2, dim=-1)
                n_in = torch.norm(net, p=2, dim=-1)
                cossim = framework.utils.cossim(net_o, net).abs().mean()
                self.log(f"activation_norm/abs_update_layer_{li}", ndiff.mean())
                self.log(f"activation_norm/in_layer_{li}", n_in.mean())
                self.log(f"activation_norm/rel_update_layer_{li}", (ndiff/n_in.clamp(min=torch.finfo(n_in.dtype).eps)).mean())
                self.log(f"activation_norm/io_cossim_{li}", cossim)

            if self.training and self.p_drop_layer > 0.0:
                net = torch.where(torch.rand_like(net_o[..., 0:1]) < self.p_drop_layer, net, net_o)
            else:
                net = net_o

        if self.use_last_state and n_prev_states > 0:
            # If we carry over the last state, save it here
            new_state = [((state[0] if state is not None else []) + [net.detach()])[-n_prev_states:]]

        if plot_cossim:
            with torch.no_grad():
                f_sample = [f.contiguous().view(-1, f.shape[-1])[:1024] for f in features]
                f_sample_all = torch.stack(f_sample, -2)
                scores = framework.utils.cossim(f_sample_all, f_sample_all).mean(0)
                self.log("feature_cossim", framework.visualize.plot.Heatmap(scores, range=(0, 1), textval=False))

                outs = F.softmax(self.gen_output(f_sample_all, target).transpose(0, 1), -1)
                scores = framework.utils.cossim(outs, outs).mean(0)
                self.log("out_dist_cossim", framework.visualize.plot.Heatmap(scores, range=(0, 1), textval=False))

                real_out = outs[:, -1]
                for i in range(outs.shape[-2] - 1):
                    self.log(f"out_diff_{i}", (outs[:, i] - real_out).norm(dim=-1, p=1).mean())

                del outs


        del features

        net = self.gen_output(net, target)
        self.iter += 1

        return net, new_state

