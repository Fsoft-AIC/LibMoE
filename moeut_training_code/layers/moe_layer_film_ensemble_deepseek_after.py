import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
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
        mid_layer_scale = weight_scale


        assert self.selection_mode in {"sigmoid", "gate"}
        assert self.perplexity_reg_mode in {"global", "time", "step", "layers_time", "standard"}
        self.new_counts_for_bias = None

        self.register_buffer("iter", torch.tensor(0, dtype=torch.int64), persistent=True)

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.get_initializer()(self.keys, std=dmodel ** -0.5 * mid_layer_scale)
        ### === Shared Layer ===
        self.n_shared_experts = 2
        self.keys_shared = torch.nn.Parameter(torch.empty(1, self.k_vec_dim, self.expert_size * self.n_shared_experts))
        self.get_initializer()(self.keys_shared, std=dmodel ** -0.5 * mid_layer_scale)
        ### ====================

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.bias_shared = torch.nn.Parameter(torch.zeros(1, self.expert_size * self.n_shared_experts))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.bias_shared = None
            self.o_bias = None

        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        ### === Shared Layer ===
        self.values_shared = torch.nn.Parameter(torch.empty(1, self.expert_size * self.n_shared_experts, self.v_dim))
        ### ====================

        sel_count = self.n_experts

        self.expert_sel = torch.nn.Parameter(torch.empty(sel_count, self.k_vec_dim))
        self.sel_bias = torch.nn.Parameter(torch.zeros(sel_count)) if sel_bias else None
        self.sel = lambda x: F.linear(x, self.expert_sel, self.sel_bias)

        self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * sel_weight_scale)

        real_size = self.size
        real_size_shared = self.n_shared_experts * self.expert_size

        self.get_initializer()(self.values, std=real_size ** -0.5 * weight_scale)
        self.get_initializer()(self.values_shared, std=real_size_shared ** -0.5 * weight_scale)

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

        #### Deepseek Setting
        self.routed_scaling_factor = 2.5
        #### ================

        #### FiLM Enssemble Setting
        self.film_ensemble = torch.nn.Parameter(torch.ones(self.v_dim), requires_grad=True)
        # self.get_initializer()(self.film_ensemble, std=self.v_dim ** -0.5)
        #### =====================

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


    def compute_scores(self, input: torch.Tensor, index: CVMMSel, is_shared: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not is_shared:
            if self.keys is not None:
                scores = cvmm(input, index, self.keys)
            if self.bias is not None:
                scores = scores + self.bias[index.raw_sel]

        elif is_shared:
            if self.keys_shared is not None:
                scores = cvmm(input, index, self.keys_shared)
            if self.bias_shared is not None:
                scores = scores + self.bias_shared[index.raw_sel]
        else:
            assert False

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
        if self.selection_mode in {"sigmoid"}:
            sel = torch.sigmoid(sel)
        elif self.selection_mode in {"gate"}:
            sel = F.softmax(sel, dim=-1)
            with torch.no_grad():
                self.log("expert_rel_perplexity_per_selection", utils.relative_perplexity(sel).mean())
        else:
            assert False

        return sel, reg_sel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, h = input.shape
        in1 = in2 = in3 = input

        sel_input = in1
        if self.selection_dropout > 0 and self.training:
            sel_input = F.dropout(sel_input, self.selection_dropout)

        sel = self.sel(sel_input)
        sel_raw = reg_sel = sel

        inv_val = float("-inf")

        if not self.activation_after_topk:
            sel, reg_sel = self.sel_activation(sel, input.shape[-2])
        sel_aux = sel

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel2 = sel.masked_fill(mask, inv_val)
        else:
            sel2 = sel

        sel_val, sel_index = self.topk(sel2, self.n_heads)
        # norm gate to sum 1
        denominator = sel_val.sum(dim=-1, keepdim=True) + 1e-20
        sel_val = sel_val / denominator
        sel_val = sel_val * self.routed_scaling_factor          # deepseek route scaling factor

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

        if reg_sel is not None:
            if self.perplexity_reg_mode in {"step", "time"}:
                self.add_perplexity_reg(reg_sel)
            elif self.perplexity_reg_mode in {"standard"}:
                self.add_perplexity_reg_standard(sel_aux, sel_index, bsz, seq_len)
            elif self.perplexity_reg > 0 and self.training:
                self.sel_hist.append(reg_sel)


        # Prepare all selection indices at once
        sel_indices = cvmm_prepare_sel2(sel_index.int())
        sel_indices_shared = cvmm_prepare_sel2(torch.zeros_like(sel_index[..., :1]))

        # Compute all scores in parallel
        scores = {
            'main': self.compute_scores(in2, sel_indices),
            'shared': self.compute_scores(in3, sel_indices_shared, is_shared=True)
        }

        # Prepare weights for all branches
        weights = {
            'main': sel_val,
            'shared': torch.ones_like(sel_val[..., :1])
        }

        # Compute outputs in parallel
        out_main = self._compute_branch_output(scores['main'], sel_indices, sel_val, self.values)
        out_shared = self._compute_branch_output(scores['shared'], sel_indices_shared, weights['shared'], self.values_shared)

        # Final output with FiLM ensemble and out_shared as beta, self.film_ensemble as gamma
        out = out_main * self.film_ensemble + out_shared

        self.layer += 1

        self.was_training = self.training
        res = out.view(*input.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        return res


    def _compute_branch_output(self, scores, indices, weights, values):
        """Helper method to compute output for each branch"""
        indices = indices.clone()
        indices.reduction_weight = weights
        indices.sel_index = indices.out_index
        indices.out_index = None
        return cvmm(scores, indices, values)


    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if self.coocurence is not None:
            coo = self.coocurence / self.coocurence.diagonal().clamp(min=1)[:, None]
            res["expert_coocurence"] = framework.visualize.plot.Heatmap(coo, xlabel="expert", ylabel="expert", textval=False)
            self.coocurence = None
        return res
