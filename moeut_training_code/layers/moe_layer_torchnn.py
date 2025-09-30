import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
from framework.layers import LoggingLayer
from framework.layers import RegularizedLayer
from framework import utils
import framework
import math
from framework.layers import OncePerIterLayer


# >>> CHANGED – simple 2-layer MLP expert (Linear-ReLU-Linear)
class MLPExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int, out_dim: int, use_bias: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=use_bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # (N, d_model) → (N, out_dim)
        return self.net(x)
    

class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):        
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        n_heads: int,
        dropout: float = 0.0,
        weight_scale: float = 1.0,
        selection_mode: str = "sigmoid",
        perplexity_reg: float = 0.0,
        perplexity_reg_mode: str = "step",
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


        assert self.selection_mode in {"sigmoid", "gate"}
        assert self.perplexity_reg_mode in {"global", "time", "step", "layers_time", "standard"}

        self.keys   = None
        self.values = None
        
        sel_count = self.n_experts

        self.expert_sel = torch.nn.Parameter(torch.empty(sel_count, self.k_vec_dim))
        self.sel_bias = torch.nn.Parameter(torch.zeros(sel_count)) if sel_bias else None

        self.sel = lambda x: F.linear(x, self.expert_sel, self.sel_bias)

        self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * sel_weight_scale)

        self.experts = nn.ModuleList([
            MLPExpert(d_model=dmodel,
                      hidden=expert_size,
                      out_dim=self.v_dim,
                      use_bias=bias)
            for _ in range(n_experts)
        ])

        self.register_buffer("iter", torch.tensor(0, dtype=torch.int64), persistent=True)
        self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm   = 0
        self.index_sel_counts_100 = 0
        self.index_sel_norm_100   = 0
        self.index_sel_counts_per_layer      = []
        self.index_sel_counts_per_layer_100  = 0
        self.coocurence = None
        self.prev_sel_oh = None
        self.register_buffer("seq", torch.arange(max(self.n_heads,
                                                     self.n_experts,
                                                     self.k_dim,
                                                     self.v_dim),
                                                 dtype=torch.long),
                             persistent=False)

    # def keys_to_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
    #     k = keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
    #     return k.permute(0, 2, 1).contiguous().view(-1, self.k_vec_dim)

    # def keys_from_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
    #     return keys.view(self.n_experts, self.expert_size, self.k_vec_dim).permute(0, 2, 1).contiguous().view(self.n_experts * self.k_vec_dim, self.expert_size)


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


    def sel_activation(self, sel: torch.Tensor, seq_len: int):
        if self.selection_mode == "sigmoid":
            sel_act = torch.sigmoid(sel)
        elif self.selection_mode == "gate":
            sel_act = F.softmax(sel, dim=-1)
        else:
            raise ValueError(f"Unknown selection mode {self.selection_mode}")
        return sel_act, sel      # (activated, raw)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input : (B, T, d_model)
        returns: (B, T, v_dim)
        """
        bsz, seq_len, _ = input.shape
        in1 = in2 = input                                        # keep identical references

        # ------------------- 1) Gating ----------------------------------
        sel_input = F.dropout(in1, self.selection_dropout, self.training) \
                    if (self.training and self.selection_dropout > 0.0) else in1
        sel_raw = reg_sel = self.sel(sel_input)                 # (B, T, n_experts)

        # --- activation / masking before top-k
        inv_val = float("-inf")
        if not self.activation_after_topk:
            sel_raw, reg_sel = self.sel_activation(sel_raw, seq_len)

        sel_aux = sel_raw                                        # keep a copy for reg later

        # (optionally) drop random experts during training
        if self.training and self.expert_dropout > 0.0:
            mask = torch.rand_like(sel_raw) < self.expert_dropout
            sel2 = sel_raw.masked_fill(mask, inv_val)
        else:
            sel2 = sel_raw

        # ------------------- 2) Top-k selection -------------------------
        sel_val, sel_index = self.topk(sel2, self.n_heads)       # each token chooses n_heads experts
        sel_val = sel_val / (sel_val.sum(dim=-1, keepdim=True) + 1e-20)

        if self.activation_after_topk:
            sel_val = torch.gather(reg_sel, -1, sel_index)       # get scores for chosen experts
            sel_val, reg_sel = self.sel_activation(sel_val, seq_len)

        # ------------------- 3) Expert computation (NEW) ----------------
        # >>> CHANGED – Re-compute token outputs WITHOUT any cvmm
        out = torch.zeros(bsz, seq_len, self.v_dim,
                          dtype=in2.dtype, device=in2.device)   # accumulator

        # Flatten once for easy indexing
        inp_flat  = in2.reshape(-1, in2.shape[-1])               # (B*T, d_model)
        out_flat  = out.reshape(-1, self.v_dim)

        for h in range(self.n_heads):
            idx_h      = sel_index[:, :, h].reshape(-1)          # (B*T,) expert ids
            weight_h   = sel_val[:, :, h].reshape(-1)            # (B*T,) mixture weights

            # For each expert, gather its tokens, run the MLP, scatter-add back
            for expert_id, expert in enumerate(self.experts):
                mask = idx_h == expert_id                        # boolean mask for this expert
                if mask.any():
                    x_e      = inp_flat[mask]                    # tokens for this expert
                    y_e      = expert(x_e)                       # (N_e, v_dim)
                    out_flat[mask] += (weight_h[mask].unsqueeze(-1) * y_e)

        out = out_flat.view(bsz, seq_len, self.v_dim)
        # ----------------------------------------------------------------

        return out

    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if self.coocurence is not None:
            coo = self.coocurence / self.coocurence.diagonal().clamp(min=1)[:, None]
            res["expert_coocurence"] = framework.visualize.plot.Heatmap(coo, xlabel="expert", ylabel="expert", textval=False)
            self.coocurence = None
        return res
