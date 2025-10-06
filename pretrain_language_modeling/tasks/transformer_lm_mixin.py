import framework
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from typing import List, Tuple, Dict, Any
from models import TransformerLanguageModel
from framework.task import task, args
from layers.transformer.relative_moe_transformer import RelativeMoeTransformerEncoderLayer
from layers.transformer.fast_rope_transformer import FastRopeTransformerEncoderLayer
from layers.transformer.full_moe_relative_attention import FullMoeRelativeAttentionCore
from layers.transformer.sut.sut_transformer_layer import SUTTransformer
from layers.transformer.sut.halting import ACTWrapper
from framework.layers.layer_with_visualization import LayerVisualizer
from layers import MoE
from framework.interfaces import Result
import os


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.trafo.context_blocks", default=1)
    parser.add_argument("-lm.trafo.test_context_blocks", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.trafo.same_length_eval", default=False)
    parser.add_argument("-lm.trafo.same_length", default=False)
    parser.add_argument("-lm.trafo.last_layer_context", default=False)
    parser.add_argument("-lm.trafo.xl_init", default=False)
    parser.add_argument("-lm.trafo.norm_input", default=False)
    parser.add_argument("-rope.rotate_fraction", default=0.5)
    parser.add_argument("-rope.base", default=10000.0)
    parser.add_argument("-pkm.n_heads", default=1)
    parser.add_argument("-moe.n_experts", default=128)
    parser.add_argument("-moe.expert_size", default=128)
    parser.add_argument("-moe.selection_mode", default="sigmoid", choice=["gate", "sigmoid"])
    parser.add_argument("-moe.perplexity_reg", default=0.0)
    parser.add_argument("-moe.perplexity_reg_mode", default="step", choice=["step", "global", "time", "layers_time", "deepseek", "standard", "jain_fairness"])
    parser.add_argument("-moe.att.perplexity_reg_mode", default="none", parser=parser.str_or_none_parser)
    parser.add_argument("-moe.activation_after_topk", default=False)
    parser.add_argument("-moe.att.expert_size", default=256)
    parser.add_argument("-moe.bias", default=False)
    parser.add_argument("-moe.sel_bias", default=False)
    parser.add_argument("-moe.dropout_factor", default=1.0)
    parser.add_argument("-moe.drop_expert", default=0.0)
    parser.add_argument("-moe.sync_distributed", default=True)
    parser.add_argument("-moe.init_scale", default=1.0)
    parser.add_argument("-moe.att.mla_attention", default=False)
    parser.add_argument("-moe.att.n_experts", default=4)
    parser.add_argument("-moe.att.enable", default=False)
    parser.add_argument("-moe.att.q_expert", default=True)
    parser.add_argument("-moe.att.k_expert", default=True)
    parser.add_argument("-moe.att.v_expert", default=True)
    parser.add_argument("-moe.att.o_expert", default=True)
    parser.add_argument("-moe.att.k", default=2)
    parser.add_argument("-moe.att.v_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-moe.att.same_sel", default=False)
    parser.add_argument("-moe.att.expert_dropout", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.selection_mode", default="sigmoid", choice=["sigmoid", "softmax", "gumbel_sigmoid"])
    parser.add_argument("-moe.att.perplexity_reg", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.k", default=2)
    parser.add_argument("-moe.att.drop_expert", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.separate_kq_sel", default=False)
    parser.add_argument("-moe.att.norm_init", default=False)
    parser.add_argument("-moe.att.dropout", default=0.0)
    parser.add_argument("-moe.att.selection_dropout", default=0.0)
    parser.add_argument("-moe.nonorm", default=False)
    parser.add_argument("-moa.cvloss", default=0.0)
    parser.add_argument("-moa.switchloss", default=0.0)
    parser.add_argument("-moa.zloss", default=0.0)
    parser.add_argument("-moa.miloss", default=0.0)
    parser.add_argument("-sut.sample_topk", default=0)
    parser.add_argument("-sut.max_relative_positions", default=64)
    parser.add_argument("-sut.drop_gate", default=0.0)
    parser.add_argument("-moe.selection_dropout", default=0.0)
    parser.add_argument("-moe.layer_std_constant", default=2.0)
    parser.add_argument("-transformer.universal.group_size", default=1)
    parser.add_argument("-transformer.universal.group_type", default="abab", choice=["abab", "aabb"])
    parser.add_argument("-transformer.embedding_scale", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-transformer.topk_value", default=32)
    parser.add_argument("-transformer.activation", default="relu", choice=["relu", "topk", "gelu", "identity", "sigmoid", "softmax"])
    parser.add_argument("-transformer.p_drop_layer", default=0.0)
    parser.add_argument("-transformer.head_projection_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.act_loss", default=0.0)
    parser.add_argument("-transformer.plot_head_details", default=False)
    parser.add_argument("-lm.trafo.force_out_norm", default=False)
    parser.add_argument("-plot.n_steps", default=-128)
    parser.add_argument("-dump_validation_plots", default="")
    parser.add_argument("-details_log_interval", default="100", parser=parser.int_or_none_parser)

@task()
class TransformerLMMixin:
    helper: framework.helpers.TrainingHelper
    VIS_DATASET_FILTER = None

    def is_preln(self) -> bool:
        return "preln" in self.helper.args.transformer.variant

    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        nx = -x
        return torch.masked_fill(x, nx <= nx.kthvalue(self.helper.args.transformer.topk_value, keepdim=True)[0], 0)

    def get_layers(self) -> List[torch.nn.Module]:
        # pyright: reportOptionalMemberAccess=false
        if self.helper.args.transformer.activation == "relu":
            activation = F.relu
        elif self.helper.args.transformer.activation == "topk":
            activation = self.topk_activation
        elif self.helper.args.transformer.activation == "identity":
            activation = lambda x: x
        elif self.helper.args.transformer.activation == "sigmoid":
            activation = torch.sigmoid
        elif self.helper.args.transformer.activation == "gelu":
            activation = F.gelu
        elif self.helper.args.transformer.activation == "softmax":
            activation = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Invalid activation: {self.helper.args.transformer.activation}")

        base_args = dict(
            d_model=self.helper.args.state_size,
            nhead=self.helper.args.transformer.n_heads,
            dim_feedforward=int(self.helper.args.state_size * self.helper.args.transformer.ff_multiplier),
            dropout=self.helper.args.dropout,
            activation=activation
        )


        extra_args = {} if not self.helper.args.transformer.variant.endswith("_gelu") else {
            "activation": F.gelu,
            "drop_expand": False
        }

        if self.helper.args.transformer.variant in {"sut_universal", "preln_sut_universal", "actsut_universal", "preln_actsut_universal"}:
            assert self.helper.args.transformer.head_projection_size is not None

            mklayer = lambda: SUTTransformer(
                d_model = self.helper.args.state_size,
                dropout = self.helper.args.dropout,
                activation_dropout = self.helper.args.dropout,
                attention_dropout= self.helper.args.moe.att.dropout,
                attn_num_expert=self.helper.args.moe.att.n_experts,
                attn_k=self.helper.args.moe.att.k,
                attn_expert_dim=self.helper.args.moe.att.expert_size,
                head_dim=self.helper.args.transformer.head_projection_size,
                preln=self.is_preln(),
                ff_expert_dim=self.helper.args.moe.expert_size,
                ff_num_expert=self.helper.args.moe.n_experts,
                ff_k=self.helper.args.pkm.n_heads,
                cvloss=self.helper.args.moa.cvloss,
                switchloss=self.helper.args.moa.switchloss,
                miloss=self.helper.args.moa.miloss,
                activation_fn=activation,
                sample_topk=self.helper.args.sut.sample_topk,
                max_relative_positions=self.helper.args.sut.max_relative_positions,
                gating_dropout=self.helper.args.sut.drop_gate,
            )

            if "act" in self.helper.args.transformer.variant:
                mklayer_old = mklayer
                mklayer = lambda: ACTWrapper(mklayer_old(), d_model=self.helper.args.state_size, act_loss=self.helper.args.transformer.act_loss)
        elif self.helper.args.transformer.variant in {"preln_rope", "rope", "preln_rope_parallel", "preln_rope_universal", "rope_universal"}:
            mklayer = lambda: FastRopeTransformerEncoderLayer(
                **base_args, **extra_args,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                preln=self.is_preln(), rotate_fraction = self.helper.args.rope.rotate_fraction,
                rope_base=self.helper.args.rope.base,
                parallel=self.helper.args.transformer.variant.endswith("_parallel"))
        elif self.helper.args.transformer.variant in {"preln_moe", "preln_moe_universal", "moe", "moe_universal",
                                                      "preln_moe_parallel"}:
            # def __init__(self, d_model, nhead, n_bins: int, bin_size: int, n_layers: int, dim_feedforward=2048,
            mklayer = lambda i: RelativeMoeTransformerEncoderLayer(
                **base_args, **extra_args, preln=self.is_preln(),
                n_layers=self.helper.args.transformer.encoder_n_layers,
                layer_idx=i,  # Add this parameter represent for layer_idx
                n_experts=self.helper.args.moe.n_experts,
                expert_size=self.helper.args.moe.expert_size,
                selection_mode=self.helper.args.moe.selection_mode,
                perplexity_reg=self.helper.args.moe.perplexity_reg,
                n_heads=self.helper.args.pkm.n_heads,
                perplexity_reg_mode=self.helper.args.moe.perplexity_reg_mode,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                activation_after_topk=self.helper.args.moe.activation_after_topk,
                sel_bias=self.helper.args.moe.sel_bias,
                bias=self.helper.args.moe.bias,
                moe_dropout_factor=self.helper.args.moe.dropout_factor,
                drop_expert=self.helper.args.moe.drop_expert,
                sync_distributed=self.helper.args.moe.sync_distributed,
                moe_init_scale=self.helper.args.moe.init_scale,
                mla_attention=self.helper.args.moe.att.mla_attention,
                moe_attention=self.helper.args.moe.att.enable,
                moe_att_n_experts=self.helper.args.moe.att.n_experts,
                moe_att_expert_dropout=self.helper.args.moe.drop_expert if self.helper.args.moe.att.drop_expert is None else self.helper.args.moe.att.drop_expert,
                moe_att_selection_mode=self.helper.args.moe.att.selection_mode,
                moe_att_ppl_reg=self.helper.args.moe.perplexity_reg if self.helper.args.moe.att.perplexity_reg is None else self.helper.args.moe.att.perplexity_reg,
                moe_att_k=self.helper.args.moe.att.k,
                q_expert=self.helper.args.moe.att.q_expert,
                k_expert=self.helper.args.moe.att.k_expert,
                v_expert=self.helper.args.moe.att.v_expert,
                o_expert=self.helper.args.moe.att.o_expert,
                v_projection_size=self.helper.args.moe.att.v_size,
                moe_att_separate_kq_sel=self.helper.args.moe.att.separate_kq_sel,
                rotate_fraction=self.helper.args.rope.rotate_fraction,
                rope_base=self.helper.args.rope.base,
                moe_att_norm_init=self.helper.args.moe.att.norm_init,
                moe_att_same_sel=self.helper.args.moe.att.same_sel,
                attention_dropout=self.helper.args.moe.att.dropout,
                moe_selection_dropout=self.helper.args.moe.selection_dropout,
                moe_att_selection_dropout=self.helper.args.moe.att.selection_dropout,
                att_perplexity_reg_mode=self.helper.args.moe.att.perplexity_reg_mode,
                log_interval=self.helper.args.details_log_interval,
                nonorm=self.helper.args.moe.nonorm,
                layer_std_constant=self.helper.args.moe.layer_std_constant,
                )

        else:
            assert False, "Invalid variant"

        if "universal" in self.helper.args.transformer.variant:
            n_uni_layers = self.helper.args.transformer.encoder_n_layers

            if n_uni_layers % self.helper.args.transformer.universal.group_size != 0:
                raise ValueError("Number of universal layers must be divisible by group size")

            if self.helper.args.transformer.universal.group_size == 1 or self.helper.args.transformer.universal.group_type == "abab":
                n_uni_layers = n_uni_layers // self.helper.args.transformer.universal.group_size
                one_block = [mklayer() for _ in range(self.helper.args.transformer.universal.group_size)]
                shared_layers = one_block * n_uni_layers
            elif self.helper.args.transformer.universal.group_type == "aabb":
                n_uni_layers = n_uni_layers // self.helper.args.transformer.universal.group_size
                shared_layers = sum([[mklayer()] * n_uni_layers for _ in range(self.helper.args.transformer.universal.group_size)], [])
            else:
                raise ValueError(f"Invalid group type: {self.helper.args.transformer.universal.group_type}")

            layers = shared_layers
        else:
            layers = [mklayer(i) for i in range(self.helper.args.transformer.encoder_n_layers)]

        return layers


    def fix_init(self, model):
        init_std = 0.02

        torch.nn.init.normal_(model.embedding.weight, 0.0, init_std)
        # torch.nn.init.normal_(model.embedding_adapter.weight, 0.0, init_std)

        initialized = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and hasattr(m, "weight"):
                torch.nn.init.normal_(m.weight, 0.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                initialized += m.bias.numel()
            if isinstance(m, (torch.nn.LayerNorm)) and m.weight is not None:
                torch.nn.init.normal_(m.weight, 1.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, MoE):
                torch.nn.init.normal_(m.keys, 0.0, init_std)
                torch.nn.init.normal_(m.values, 0.0, init_std)
                if m.expert_sel is not None:
                    torch.nn.init.normal_(m.expert_sel, 0.0, init_std)
                    m.fix_expert_sel_init()
                    initialized += m.expert_sel.numel()
                initialized += m.keys.numel() + m.values.numel()
            if isinstance(m, FullMoeRelativeAttentionCore):
                for p in m.parameters():
                    torch.nn.init.normal_(p, 0.0, init_std)
                    initialized += p.numel()

            if isinstance(m, FullMoeRelativeAttentionCore):
                for s in m.selections.values():
                    m.renorm_keep_std(s)


        print(f"Reinitialized {initialized/self.n_weights*100:.3f}% weights")

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        if (self.VIS_DATASET_FILTER is None) or (name in self.VIS_DATASET_FILTER):
            self.validation_started_on = name
            self.validation_step = 0

        res = super().validate_on_name(name)
        return res

    def create_model(self) -> torch.nn.Module:
        self.validation_started_on = None
        # pyright: reportOptionalMemberAccess=false
        tlayers = self.get_layers()

        model = TransformerLanguageModel(
            len(self.train_set.vocabulary), self.helper.args.embedding_size,
            self.helper.args.state_size, self.helper.args.dropout,
            layers=tlayers, n_prev_states=self.helper.args.lm.trafo.context_blocks,
            n_prev_states_test=self.helper.args.lm.trafo.test_context_blocks,
            same_length_eval=self.helper.args.lm.trafo.same_length_eval,
            p_drop_layer=self.helper.args.transformer.p_drop_layer,
            same_length=self.helper.args.lm.trafo.same_length,
            use_last_state=self.helper.args.lm.trafo.last_layer_context,
            norm_before_output=self.is_preln() or self.helper.args.lm.trafo.force_out_norm,
            norm_input=self.helper.args.lm.trafo.norm_input,
            cross_layer_state="actsut" in self.helper.args.transformer.variant,
            log_interval=self.helper.args.details_log_interval,
            )

        self.n_weights = sum(p.numel() for p in model.parameters())
        self.n_weights_model = sum(p.numel() for p in model.unique_layers.parameters() if p.requires_grad)
        self.n_attention_weights = sum(p.numel() for n, p in model.unique_layers.named_parameters() if p.requires_grad and ("attention" in n or "attn" in n))

        weight_info = {
            "n_model_weights": self.n_weights_model,
            "n_attention_weights": self.n_attention_weights,
            "n_non_attnetion_weights": self.n_weights_model - self.n_attention_weights,
            "attention_precent": self.n_attention_weights / self.n_weights_model,
        }

        print("Weight info:")
        for k, v in weight_info.items():
            print(f"  {k}: {v}")
        self.helper.log(weight_info)

        with torch.no_grad():
            if self.is_preln():
                model.embedding_scale = 1.0

            if self.helper.args.lm.trafo.xl_init:
                self.fix_init(model)

            if self.helper.args.transformer.embedding_scale is not None:
                model.embedding_scale = self.helper.args.transformer.embedding_scale

        self.visualizer = LayerVisualizer(model, {
            "mha.plot_head_details": self.helper.args.transformer.plot_head_details,
            "mha.no_pos_vs_content": True
        })

        self.input_history = []
        return model

    def get_steplabels(self, data: Dict[str, torch.Tensor]) -> List[str]:
        out = self.train_set.vocabulary(data["data"][:, 0].cpu().numpy().tolist())
        inp = [self.train_set.vocabulary(x[:-1].cpu().numpy().tolist()) for x in self.input_history] + [out]
        return sum(inp, [])[:-1], out[1:]

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> Tuple[Result, Dict[str, Any]]:
        plot_now = ((ubatch == 0) and (self.helper.args.debug_plot_interval is not None) and \
                   ((self.helper.state.iter % self.helper.args.debug_plot_interval) == 0) and self.model.training)

        is_dumping = self.validation_started_on and self.helper.args.dump_validation_plots

        if plot_now or is_dumping:
            inp, outp = self.get_steplabels(data)
            params = {"steplabel": inp, "target_labels": outp}
            if self.helper.args.plot.n_steps:
                params["n_steps"] = self.helper.args.plot.n_steps

            self.visualizer.prepare(params)

        if ubatch == 0 and self.helper.args.lm.trafo.context_blocks > 0:
            if len(self.input_history) >= self.helper.args.lm.trafo.context_blocks:
                self.input_history.pop(0)
            self.input_history.append(data["data"][:, 0])

        res, plots = super().run_model(data, ubatch)

        if plot_now or is_dumping:
            plots.update({f"activations/{k}": v for k, v in self.visualizer.plot().items()})

        if is_dumping:
            os.makedirs(self.helper.args.dump_validation_plots, exist_ok=True)
            torch.save(plots, f"{self.helper.args.dump_validation_plots}/{self.validation_started_on}_{self.validation_step:04d}.pth")
            self.validation_step += 1

        return res, plots

