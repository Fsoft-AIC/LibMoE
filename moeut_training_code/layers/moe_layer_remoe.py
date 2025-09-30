import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
from einops import rearrange, repeat, reduce, pack, unpack
from framework.layers import LoggingLayer
from framework.layers import RegularizedLayer
from framework import utils
import framework
import math
from framework.layers import OncePerIterLayer
from layers import cvmm, cvmm_prepare_sel
from layers.cvmm import CVMMSel, cvmm_prepare_sel2


class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int,
                 dropout: float = 0, weight_scale: float = 1.0,
                 selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
                 perplexity_reg_mode: str="step",
                 activation_after_topk: bool = False,
                 activation = lambda x: F.relu(x, inplace=True),
                 sel_bias: bool = False,
                 bias: bool = False,
                 v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 sync_distributed: bool = False,
                 selection_dropout: float = 0.0,
                 log_interval: Optional[int] = 100,
                 ):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.perplexity_reg = perplexity_reg
        self.k_vec_dim = self.k_dim
        self.n_heads = n_heads
        self.perplexity_reg_mode = perplexity_reg_mode
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.weight_scale = weight_scale
        self.layer = 0
        self.initalized = False
        self.was_training = True
        self.expert_dropout = expert_dropout
        self.reg_counts = 0
        self.sync_distributed = sync_distributed and torch.distributed.is_initialized()
        self.record_all_expert_sel_counts = False
        self.selection_dropout = selection_dropout
        self.log_interval = log_interval

        self.coocurence = None
        self.prev_sel_oh = None

        sel_weight_scale = weight_scale
        mid_layer_scale =  weight_scale


        # === ReMoE CHANGE >>> recognise ReLU routing
        assert selection_mode in {"sigmoid", "gate", "relu"}
        self.selection_mode = selection_mode

        if self.selection_mode == "relu":
            # target sparsity (1 – k/E)
            self.target_sparsity = 1.0 - (self.n_heads / self.n_experts)
            self.balance_loss_coef = 1e-2  # Start with a reasonable coefficient
            self.moe_relu_l1_reg_coeff_multiplier = 1.2
            self.moe_relu_sparsity = 0.0                                 # runtime tracker
        # <<< ReMoE CHANGE ===

        assert self.perplexity_reg_mode in {"global", "time", "step", "layers_time", "standard"}
        self.new_counts_for_bias = None

        self.register_buffer("iter", torch.tensor(0, dtype=torch.int64), persistent=True)

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.get_initializer()(self.keys, std=dmodel ** -0.5 * mid_layer_scale)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.o_bias = None

        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))

        sel_count = self.n_experts

        self.expert_sel = torch.nn.Parameter(torch.empty(sel_count, self.k_vec_dim))
        self.sel_bias = torch.nn.Parameter(torch.zeros(sel_count)) if sel_bias else None

        self.sel = lambda x: F.linear(x, self.expert_sel, self.sel_bias)

        self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * sel_weight_scale)

        real_size = self.size

        self.get_initializer()(self.values, std=real_size ** -0.5 * weight_scale)
        self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0

        self.index_sel_counts_100 = 0
        self.index_sel_norm_100 = 0
        self.index_sel_counts_per_layer = []
        self.index_sel_counts_per_layer_100 = 0

        self.sel_count_log = None

        self.all_expert_sel_counts = []
        self.all_expert_sel_soft = []

        self.register_buffer("kv_sel_counts", torch.zeros(self.n_experts, self.expert_size), persistent=False)
        self.register_buffer("kv_sel_counts_100", torch.zeros_like(self.kv_sel_counts))

        self.register_buffer("seq", torch.arange(max(self.n_heads, self.n_experts, self.k_dim, self.v_dim), dtype=torch.long), persistent=False)

    def keys_to_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        k = keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
        return k.permute(0, 2, 1).contiguous().view(-1, self.k_vec_dim)

    def keys_from_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        return keys.view(self.n_experts, self.expert_size, self.k_vec_dim).permute(0, 2, 1).contiguous().view(self.n_experts * self.k_vec_dim, self.expert_size)


    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())


    def fix_expert_sel_init(self):
        with torch.no_grad():
            self.renorm_keep_std(self.expert_sel, dim=1)

    def get_initializer(self):
        return torch.nn.init.normal_

    def sparse_matmul(self, indices: torch.Tensor, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.embedding_bag(indices, weight.type_as(values), per_sample_weights=values, mode="sum", sparse=False)

    def ani(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        chunk_size = 32

        xnorm = F.normalize(x, 2, dim=-1)

        accu = 0
        for i in range(0, x.shape[0], chunk_size):
            a = xnorm[i: i + chunk_size]
            sims = xnorm @ a.T
            sims[i : i + chunk_size].fill_diagonal_(0)
            accu += sims.sum()

        return accu / (x.shape[0] * (x.shape[0] - 1))

    def log_expert_sel_usage(self, prefix: str, channel_sel_counts: torch.Tensor):
        sel_nonzero = (channel_sel_counts != 0).type(torch.float).sum(axis=-1) / self.expert_size
        self.log(f"{prefix}/mean", sel_nonzero.mean())
        self.log(f"{prefix}/min", sel_nonzero.min())
        self.log(f"{prefix}/max", sel_nonzero.max())


    def pre_train_forward(self):
        self.prev_sel_oh = None

        if self.training and not self.was_training:
            sorted_counts = self.index_sel_counts.sort(descending=True).values
            self.log("test_exert_channel_usage", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)

        self.layer = 0
        if self.sel_hist:
            self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0
        self.reg_counts = 0
        self.index_sel_counts_per_layer = []

    def before_loss(self):
        if self.sel_hist:
            # Concatenate against time dimension. Important for the within-batch regularization
            sel = torch.stack(self.sel_hist, 1)
            self.add_perplexity_reg(sel)

            self.sel_hist = []


        if self.training and len(self.index_sel_counts_per_layer) > 1:
            index_sel_counts_per_layer = torch.stack(self.index_sel_counts_per_layer, dim=0)
            self.index_sel_counts_per_layer = []

            if torch.is_tensor(self.index_sel_counts_per_layer_100) and self.index_sel_counts_per_layer_100.shape != index_sel_counts_per_layer.shape:
                # The number of layers changed
                if self.index_sel_counts_per_layer_100.shape[0] > index_sel_counts_per_layer.shape[0]:
                    # self.index_sel_counts_per_layer_100 = self.index_sel_counts_per_layer_100[:index_sel_counts_per_layer.shape[0]]
                    index_sel_counts_per_layer = F.pad(index_sel_counts_per_layer, [0, 0, 0, self.index_sel_counts_per_layer_100.shape[0] - index_sel_counts_per_layer.shape[0]])
                else:
                    self.index_sel_counts_per_layer_100 = F.pad(self.index_sel_counts_per_layer_100, [0, 0, 0, index_sel_counts_per_layer.shape[0] - self.index_sel_counts_per_layer_100.shape[0]])

            self.index_sel_counts_per_layer_100 += index_sel_counts_per_layer

            if self.iter % self.log_interval == 0:
                index_sel_counts_per_layer_100 = framework.utils.distributed_ops.reduce_any(self.index_sel_counts_per_layer_100)
                index_sel_counts_per_layer_100 = index_sel_counts_per_layer_100.float()
                index_sel_counts_per_layer_100 /= index_sel_counts_per_layer_100.sum(-1, keepdim=True)

                self.log("moe_per_layer_100", framework.visualize.plot.Heatmap(index_sel_counts_per_layer_100, xlabel="expert", ylabel="layer", textval=False), drop_old=True)

                total = index_sel_counts_per_layer_100.sum(-1, keepdim=True)
                pairwise_overlap = torch.min(index_sel_counts_per_layer_100.unsqueeze(0), index_sel_counts_per_layer_100.unsqueeze(1)).sum(-1)
                pairwise_overlap = pairwise_overlap / total

                self.log("layer_sel_similarity_100", framework.visualize.plot.Heatmap(pairwise_overlap, xlabel="layer", ylabel="layer", textval=False), drop_old=True)

                self.log("universal_score", pairwise_overlap.mean())
                self.log("universal_score_optimist", pairwise_overlap.max(-1).values.mean())


                self.index_sel_counts_per_layer_100 = 0

        if self.index_sel_norm > 0:
            if self.training and self.log_interval is not None:
                with torch.no_grad():
                    self.index_sel_counts_100 = self.index_sel_counts_100 + self.index_sel_counts
                    self.index_sel_norm_100 = self.index_sel_norm_100 + self.index_sel_norm

                    if self.iter % self.log_interval == 0:
                        self.log("usag_rel_perplexity_all_layers", utils.relative_perplexity(self.index_sel_counts / self.index_sel_norm))
                        self.log("dead_expert_proportion_all_layers", (self.index_sel_counts == 0).float().sum() / self.n_experts)

                        if self.sel_bias is not None:
                            self.log("sel_bias_min", self.sel_bias.detach().min())
                            self.log("sel_bias_max", self.sel_bias.detach().max())

                        index_sel_counts_100 = framework.utils.distributed_ops.reduce_any(self.index_sel_counts_100)
                        index_sel_norm_100 = framework.utils.distributed_ops.reduce_any(self.index_sel_norm_100)
                        norm_cnt = index_sel_counts_100 / index_sel_norm_100
                        self.log("usag_rel_perplexity_100", utils.relative_perplexity(norm_cnt))
                        self.log("dead_expert_proportion_100", (index_sel_counts_100 == 0).float().sum() / self.n_experts)

                        sorted_counts = index_sel_counts_100.sort(descending=True).values
                        self.log("usage_counts_100", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)


                        self.index_sel_counts_100 = 0
                        self.index_sel_norm_100 = 0

                        self.log("ani/keys", self.ani(self.keys_to_logical_order(self.keys)))
                        self.log("ani/values", self.ani(self.values.flatten(0, -2)))
                        if self.expert_sel is not None:
                            self.log("ani/expert_sel", self.ani(self.expert_sel.T))

        if self.training:
            self.iter += 1

    def topk(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.topk(k, dim=-1, sorted=False)

    def logsoftmax_of_history(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate calculating logsumexp over a bigger batch than the current one. Will have stale values, but that
        # should not matter much later in training.
        return F.log_softmax(x, dim=-1)

    def add_perplexity_reg(self, sel: torch.Tensor):
        sync_distributed = self.sync_distributed and (self.perplexity_reg_mode == "global")

        if self.perplexity_reg_mode in {"time", "layers_time"}:
            sel = sel.flatten(1, -2)
        elif self.perplexity_reg_mode == "global":
            sel = sel.flatten(0, -2)
        elif self.perplexity_reg_mode == "step":
            sel = sel.flatten(0, -2).unsqueeze(-2)
        else:
            raise ValueError(f"Unknown perplexity_reg_mode: {self.perplexity_reg_mode}")

        sel = sel.float()

        # Note: sel are raw logits, no matter what activation is used
        if self.perplexity_reg > 0:
            sel_d = self.logsoftmax_of_history(sel)
            sel_d = framework.utils.distributed_ops.log_mean(sel_d, -2, sync_distributed)
            loss = lambda: self.perplexity_reg * ( - utils.entropy_l(sel_d).mean())

            self.add_reg(loss, "moe")

    def add_perplexity_reg_standard(self, sel_aux: torch.Tensor, sel_index: torch.Tensor, bsz: int, seq_len: int):
        sel_aux_scaled = sel_aux / sel_aux.sum(-1).unsqueeze(-1)
        aux_loss = torch.zeros(bsz, self.n_experts, device=sel_aux.device)
        aux_loss.scatter_add_(1, sel_index.view(bsz, seq_len * self.n_heads), torch.ones(bsz, seq_len * self.n_heads, device=sel_aux.device)).div_(seq_len * self.n_heads / self.n_experts)

        if self.perplexity_reg > 0:
            loss = lambda: self.perplexity_reg * (aux_loss * sel_aux_scaled.mean(dim = 1)).sum(dim = 1).mean()

            self.add_reg(loss, "moe")


    def compute_scores(self, input: torch.Tensor, index: CVMMSel) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is not None:
            scores = cvmm(input, index, self.keys)

        if self.bias is not None:
            scores = scores + self.bias[index.raw_sel]


        scores = self.activation(scores)


        plot_training = self.train and self.log_interval is not None and self.iter % self.log_interval == 0
        if plot_training:
            with torch.no_grad():
                gt0 = (scores > 0).float()
                gt0_s = gt0.sum()

                if plot_training:
                    self.log("relu_pass_rate", gt0_s / scores.numel())

        return scores

    def sel_activation(self, sel: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_sel = sel
        # === ReMoE CHANGE >>>
        if self.selection_mode == "relu":
            sel = F.relu(sel)
        # <<< ReMoE CHANGE ===
        elif self.selection_mode in {"sigmoid"}:
            sel = torch.sigmoid(sel)
        elif self.selection_mode in {"gate"}:
            sel = F.softmax(sel, dim=-1)
            with torch.no_grad():
                self.log("expert_rel_perplexity_per_selection", utils.relative_perplexity(sel).mean())
        else:
            assert False
        return sel, reg_sel


    # === ReMoE CHANGE >>>  adaptive λ update (Eq. 7) and sparsity tracker
    def _update_relu_stats(self, relu_out: torch.Tensor):
        with torch.no_grad():
            routing_map = relu_out > 0
            sparsity = 1.0 - routing_map.sum().float() / routing_map.numel()
            self.moe_relu_sparsity = sparsity.item()  # Store as scalar

            # Adaptive lambda update: increase lambda if sparsity is too low, decrease if too high
            if sparsity < self.target_sparsity:
                self.balance_loss_coef *= self.moe_relu_l1_reg_coeff_multiplier
            else:
                self.balance_loss_coef /= self.moe_relu_l1_reg_coeff_multiplier

            # Clamp lambda to reasonable bounds
            self.balance_loss_coef = torch.clamp(torch.tensor(self.balance_loss_coef), 1e-6, 1e2).item()
    # <<< ReMoE CHANGE ===

    def _balance_loss(self, sel_index: torch.Tensor, sel_val: torch.Tensor) -> torch.Tensor:
        """
        Fedus-style load-balancing (density proxy × actual density) scaled by E².
        sel_index : (B,T,H)  indices of chosen experts
        sel_val   : (B,T,H)  un-normalised positive weights (post-ReLU)
        """
        with torch.no_grad():
            # Create proxy density by averaging selected values per expert
            proxy_density = torch.zeros(self.n_experts, device=sel_val.device, dtype=sel_val.dtype)
            proxy_density.index_add_(0, sel_index.flatten(), sel_val.flatten())
            proxy_density = proxy_density / (sel_index.numel() / self.n_experts)  # Normalize

            # Create actual density from one-hot encoding
            one_hot = F.one_hot(sel_index, self.n_experts).float()
            actual_density = one_hot.mean(dim=(0, 1, 2))                # (E,)

        return (proxy_density * actual_density).sum() * (self.n_experts ** 2)

    def _apply_l1_load_balancing_loss(self, relu_out: torch.Tensor, routing_map: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        """
        Apply proper L1 load-balancing loss following the original ReMoE implementation.
        This uses the same formula as switch load balancing loss but applied to ReLU outputs.
        """
        # Calculate tokens per expert (actual density)
        tokens_per_expert = routing_map.sum(dim=(0, 1, 2))  # (E,)

        # Calculate average activation per expert (proxy density)
        avg_activation_per_expert = relu_out.mean(dim=(0, 1, 2))  # (E,)

        # Total tokens
        total_tokens = bsz * seq_len

        # Normalize densities
        actual_density = tokens_per_expert.float() / total_tokens  # (E,)
        proxy_density = avg_activation_per_expert  # Already normalized by mean

        # Load balancing loss: sum of (proxy_density * actual_density) scaled by E^2
        loss = (proxy_density * actual_density).sum() * (self.n_experts ** 2)

        return loss

    def _relu_routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ReLU routing function that applies ReLU and selects all non-zero experts.
        Unlike top-k routing, this doesn't limit to k experts but uses all positive outputs.
        """
        # Apply ReLU to get positive activations
        relu_out = F.relu(logits)

        # Create routing map (boolean mask for non-zero activations)
        routing_map = relu_out > 0

        # For compatibility with the rest of the code, we still need to provide
        # sel_val and sel_index, but they represent all non-zero activations
        # We'll use the original approach but without top-k limitation for ReLU

        return relu_out, routing_map

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, h = input.shape
        in1 = in2 = input

        sel_input = in1
        if self.selection_dropout > 0 and self.training:
            sel_input = F.dropout(sel_input, self.selection_dropout)

        sel = self.sel(sel_input)   # logits
        sel_raw = reg_sel = sel

        inv_val = float("-inf")

        if not self.activation_after_topk:
            sel, reg_sel = self.sel_activation(sel, input.shape[-2])    # after relu

        sel_aux = sel

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel2 = sel.masked_fill(mask, inv_val)
        else:
            sel2 = sel

        # === ReMoE CHANGE >>> Use proper ReLU routing
        if self.selection_mode == "relu":
            sel_val_full = F.relu(sel2)
            routing_map = sel_val_full > 0
            sel_val, sel_index = self.topk(sel_val_full, self.n_heads)
        else:
            sel_val, sel_index = self.topk(sel2, self.n_heads)
            denominator = sel_val.sum(dim=-1, keepdim=True) + 1e-6
            sel_val = sel_val / denominator
        # <<< ReMoE CHANGE ===

        record_counts_now = (self.training and self.iter % 10 == 0) or (not self.training) or (self.record_all_expert_sel_counts)

        if not self.training:
            sel_index_flat = sel_index.flatten(end_dim=-2)
            if self.coocurence is None:
                self.coocurence = torch.zeros([self.n_experts, self.n_experts], device=sel_index_flat.device, dtype=torch.long)

            for h1 in range(self.n_heads):
                for h2 in range(self.n_heads):
                    ind_flat = sel_index_flat[..., h1] * self.n_experts + sel_index_flat[..., h2]
                    values = torch.tensor([1], device=self.coocurence.device, dtype=self.coocurence.dtype).expand_as(ind_flat)
                    # values = sel_val[..., h2].flatten()
                    self.coocurence.flatten().put_(ind_flat, values, accumulate=True)
                    # self.coocurence[sel_index_flat[..., h1], sel_index_flat[..., h2]] += 1

        if record_counts_now:
            reg_counts = F.one_hot(sel_index, self.n_experts).type_as(input)

            with torch.no_grad():
                sel_counts = reg_counts.flatten(end_dim=-2).sum(0)
                cnt = sel_index.nelement()

                # p_expert_sel = sel_counts / cnt

                self.index_sel_counts = self.index_sel_counts + sel_counts
                self.index_sel_norm = self.index_sel_norm + cnt

                if self.record_all_expert_sel_counts:
                    softcnt = torch.zeros_like(sel_counts, dtype=sel_val.dtype)
                    softcnt.index_add_(0, sel_index.flatten(), sel_val.flatten())

                    self.all_expert_sel_soft.append(softcnt)
                    self.all_expert_sel_counts.append(sel_counts)

                if self.training and self.log_interval is not None:
                    self.index_sel_counts_per_layer.append(sel_counts)

                    if self.iter % self.log_interval == 0:
                        self.log("min_sel_score", sel_val.min(dim=-1).values.mean())
                        self.log("max_sel_score", sel_val.max(dim=-1).values.mean())

                        sel_oh = F.one_hot(sel_index, self.n_experts).sum(-2).bool()
                        if self.prev_sel_oh is not None and self.training:
                            self.log(f"layer_sel_overlap_{self.layer}", ((self.prev_sel_oh & sel_oh).sum(-1).float() / self.n_heads).mean())

                        self.prev_sel_oh = sel_oh

        # === ReMoE CHANGE >>>  add ReLU-specific regularisation & logging
        if self.selection_mode == "relu":
            # 1) sparsity tracking + λ update
            relu_full = F.relu(sel_raw)
            routing_map_full = relu_full > 0
            self._update_relu_stats(relu_full)

            # 2) proper L1 load-balancing loss (following original ReMoE)
            l1_loss = self._apply_l1_load_balancing_loss(relu_full, routing_map_full, bsz, seq_len)
            self.add_reg(lambda: self.balance_loss_coef * l1_loss, "moe")

            # 3) optional auxiliary diagnostics (logged but not back-prop'd)
            with torch.no_grad():
                bal_loss = self._balance_loss(sel_index, sel_val)
                self.log("remoe/balance_loss", bal_loss)
                self.log("remoe/sparsity",     self.moe_relu_sparsity)
                self.log("remoe/λ",            self.balance_loss_coef)
                self.log("remoe/l1_loss",      l1_loss.detach())
        # <<< ReMoE CHANGE ===


        # sel_indices = [cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts) for h in range(sel_index.shape[-1])]
        sel_indices = cvmm_prepare_sel2(sel_index.int())

        scores = self.compute_scores(in2, sel_indices)

        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        out = cvmm(scores, sel_indices, self.values)

        self.layer += 1

        self.was_training = self.training
        res = out.view(*input.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        return res

    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if self.coocurence is not None:
            coo = self.coocurence / self.coocurence.diagonal().clamp(min=1)[:, None]
            res["expert_coocurence"] = framework.visualize.plot.Heatmap(coo, xlabel="expert", ylabel="expert", textval=False)
            self.coocurence = None
        return res