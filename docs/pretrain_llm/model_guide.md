# MoE Model Guide (Pretrain LLM)

Different MoE methods live in `pretrain_language_modeling/layers`. The core sparse batched matmul used by all MoE variants is implemented with Triton in `pretrain_language_modeling/layers/cvmm.py`.

The constructor of the reference MoE (`pretrain_language_modeling/layers/moe_layer.py`) accepts:
```python
def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int,
             dropout: float = 0, weight_scale: float = 1.0,
             selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
             perplexity_reg_mode: str = "step",
             activation_after_topk: bool = False,
             activation = lambda x: F.relu(x, inplace=True),
             sel_bias: bool = False,
             bias: bool = False,
             v_dim: Optional[int] = None,
             expert_dropout: float = 0.0,
             sync_distributed: bool = False,
             selection_dropout: float = 0.0,
             log_interval: Optional[int] = 100):
    ...
```

Argument notes:
- `dmodel` — input feature dimension
- `n_experts` — number of experts
- `expert_size` — per‑expert hidden size
- `n_heads` — top‑k per token (k)
- `dropout` — residual dropout around MoE block
- `weight_scale` — init scale factor for parameters
- `selection_mode` — `"sigmoid"` or `"gate"` (softmax across experts). Use `"gate"` for softmax routing.
- `perplexity_reg` — load‑balancing strength (typical: 0.001)
- `perplexity_reg_mode` — `"step"`, `"time"`, `"layers_time"`, `"global"`, or `"standard"`
- `activation_after_topk` — apply router activation before or after top‑k
- `activation` — expert nonlinearity (default ReLU)
- `sel_bias` / `bias` — optional biases for router / experts
- `v_dim` — output feature dim (defaults to `dmodel`)
- `expert_dropout` — drop whole experts during routing (e.g., 0.05)
- `sync_distributed` — synchronize stats for distributed training
- `selection_dropout` — dropout on router input/logits
- `log_interval` — frequency for debug/usage metrics

The forward function signature:
```python
def forward(self, input: torch.Tensor) -> torch.Tensor
```
The forward pass returns only the output tensor. Load‑balancing/entropy regularization is added via the `RegularizedLayer` interface inside the layer (see `add_perplexity_reg`), and your training loop/framework collects and adds it to the main loss.

The Triton kernels JIT‑compile on first use during the forward/backward calls.

## Example

```python
from layers import MoE

moe_layer = MoE(d_model, n_experts, expert_size, k).cuda()
y = moe_layer(x)  # y: [B, T, dmodel]
```

## How Layer Selection Works

- The chosen MoE implementation is loaded dynamically based on the `MOE_TYPE` environment variable.
- `pretrain_language_modeling/layers/__init__.py` exposes `MoE` by importing `layers.${MOE_TYPE}.MoE`.
- Default is `MOE_TYPE=moe_layer`.

Example:
```bash
export MOE_TYPE="moe_layer_xmoe"   # picks pretrain_language_modeling/layers/moe_layer_xmoe.py
```

## Adding a New MoE

1) Create a new file in `pretrain_language_modeling/layers`, for example `my_moe.py`, that defines a class named `MoE` with the same constructor and forward signature as the reference implementation.

2) Implement selection and scoring using the shared Triton utilities:
```python
from typing import Optional, Tuple
import torch, torch.nn.functional as F
from framework.layers import LoggingLayer, RegularizedLayer, OncePerIterLayer
from .cvmm import cvmm, cvmm_prepare_sel2, CVMMSel

class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int, 
                 dropout: float = 0.0, weight_scale: float = 1.0,
                 selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
                 perplexity_reg_mode: str = "step", activation_after_topk: bool = False,
                 activation = lambda x: F.relu(x, inplace=True),
                 sel_bias: bool = False, bias: bool = False,
                 v_dim: Optional[int] = None, expert_dropout: float = 0.0,
                 sync_distributed: bool = False, selection_dropout: float = 0.0,
                 log_interval: Optional[int] = 100):
        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts, self.expert_size, self.n_heads = n_experts, expert_size, n_heads
        # Expert parameters
        self.keys = torch.nn.Parameter(torch.empty(n_experts, self.k_dim, expert_size))
        self.values = torch.nn.Parameter(torch.empty(n_experts, expert_size, self.v_dim))
        # Router projection (you can customize)
        self.sel_bias = torch.nn.Parameter(torch.zeros(n_experts)) if sel_bias else None
        self.expert_sel = torch.nn.Parameter(torch.empty(n_experts, self.k_dim))
        self.sel = lambda x: F.linear(x, self.expert_sel, self.sel_bias)
        torch.nn.init.normal_(self.expert_sel, std=self.k_dim ** -0.5 * weight_scale)
        torch.nn.init.normal_(self.keys, std=dmodel ** -0.5 * weight_scale)
        torch.nn.init.normal_(self.values, std=(n_experts * expert_size) ** -0.5 * weight_scale)
        # Regularization config
        self.selection_mode = selection_mode
        self.perplexity_reg = perplexity_reg
        self.perplexity_reg_mode = perplexity_reg_mode
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.expert_dropout = expert_dropout
        self.selection_dropout = selection_dropout
        self.log_interval = log_interval

    def sel_activation(self, sel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # `gate` == softmax router; `sigmoid` is independent routing
        reg_sel = sel
        if self.selection_mode == "gate":
            sel = F.softmax(sel, dim=-1)
        elif self.selection_mode == "sigmoid":
            sel = torch.sigmoid(sel)
        else:
            raise ValueError("selection_mode must be 'gate' or 'sigmoid'")
        return sel, reg_sel

    def compute_scores(self, x: torch.Tensor, idx: CVMMSel) -> torch.Tensor:
        scores = cvmm(x, idx, self.keys)
        return self.activation(scores)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1) Router logits over experts
        sel_inp = F.dropout(input, self.selection_dropout, training=self.training) if self.selection_dropout > 0 else input
        sel = self.sel(sel_inp)
        sel_raw = sel

        # 2) Apply router activation before/after Top‑K
        if not self.activation_after_topk:
            sel, _ = self.sel_activation(sel)

        # 3) Expert Dropout and Top‑K (k = n_heads)
        if self.training and self.expert_dropout > 0:
            sel = sel.masked_fill(torch.rand_like(sel) < self.expert_dropout, float("-inf"))
        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)
        sel_val = sel_val / (sel_val.sum(dim=-1, keepdim=True) + 1e-20)

        if self.activation_after_topk:
            sel_val = torch.gather(sel_raw, -1, sel_index)
            sel_val, _ = self.sel_activation(sel_val)

        # 4) Prepare sparse selection for Triton kernels
        sel_idx = cvmm_prepare_sel2(sel_index.int())

        # 5) Compute expert scores and aggregate outputs
        scores = self.compute_scores(input, sel_idx)
        sel_idx = sel_idx.clone()
        sel_idx.reduction_weight = sel_val
        sel_idx.sel_index = sel_idx.out_index
        sel_idx.out_index = None
        out = cvmm(scores, sel_idx, self.values)
        return out.view(*input.shape[:-1], self.v_dim)
```

3) Tell the framework to use it by setting `MOE_TYPE` before training:
```bash
export MOE_TYPE="my_moe"
```

## Key Utilities (`layers/cvmm.py`)

- `cvmm(x, sel, w)`: sparse batched matmul over selected experts. Shapes:
  - `x`: `[..., K]` input features
  - `w`: `[n_experts, K, N]` or `[n_experts, N, K]` depending on usage
  - `sel`: either logits over experts or a prepared `CVMMSel` (see below)
- `cvmm_prepare_sel2(sel_index) -> CVMMSel`: packs top‑k expert indices for Triton kernels; supports reduction weights via `CVMMSel.reduction_weight`.
- `CVMMSel`: carries `raw_sel`, sorted `sel`, `sel_index`, optional `out_index`, and optional `reduction_weight`.

## Shapes and Data Flow

- Input: `[B, T, dmodel]`
- Router produces expert logits per token: `[B, T, n_experts]`
- Top‑K over experts per token with `k = n_heads`
- Keys: `[n_experts, dmodel, expert_size]` → produces per‑expert hidden `[B, T, expert_size]`
- Values: `[n_experts, expert_size, v_dim]` → projects back to model dim; default `v_dim = dmodel`
- Output: `[B, T, v_dim]`

## Built‑in Variants

- `moe_layer`: vanilla MoE with linear router and ReLU expert MLPs.
- `moe_layer_xmoe`: cosine router variant (X-MoE style) with temperature.
- `moe_layer_deepseekv2`, `moe_layer_deepseekv3`: DeepSeek‑style variants with shared keys/values and additional regularization.
- `moe_layer_remoe`, `moe_layer_plus_plus`, `moe_layer_tc_moe`: other research variants you can use as templates.

Select one via `MOE_TYPE`, e.g. `export MOE_TYPE="moe_layer_xmoe"`.

## Regularization and Logging

- Load‑balancing/entropy regularization is added inside the layer via `add_perplexity_reg` or `add_perplexity_reg_standard` depending on `perplexity_reg_mode`.
- The training framework aggregates these from all `RegularizedLayer`s; you do not need to return a separate loss from `forward`.
- `log_interval` controls periodic metrics (expert usage, overlap, ANI of weights, etc.).

## Where It’s Used

- MoE is plugged into the Transformer MLP in `pretrain_language_modeling/layers/transformer/relative_moe_transformer.py` via `pkm = MoE(...)`.
- The encoder is instantiated from `pretrain_language_modeling/tasks/transformer_lm_mixin.py` and `pretrain_language_modeling/models/transformer_language_model.py`.

## Quick Test

```python
import torch
from layers import get_moe_type

MoEClass = get_moe_type("moe_layer")
m = MoEClass(dmodel=512, n_experts=16, expert_size=64, n_heads=2).cuda()
x = torch.randn(2, 8, 512, device="cuda")
y = m(x)
print(y.shape)  # torch.Size([2, 8, 512])
```

## Common Pitfalls

- Use `selection_mode="gate"` for softmax gating. The string `"softmax"` is not accepted by MoE routers; use `"gate"` instead.
- `forward` returns only the output tensor; regularization is handled via the `RegularizedLayer` interface.
- Ensure shapes match the expected `[n_experts, dmodel, expert_size]` and `[n_experts, expert_size, v_dim]` layouts when calling `cvmm`.
- Triton kernels JIT‑compile on first use and require a compatible GPU/driver; the first pass may take longer.


# Hyperparameters

To modify the hyperparameters of the MoE layer, you can modify the sweep yaml file for each experiment in `pretrain_language_modeling/sweeps`. 


## Useful built-in arguments

- `-task`: which task to use. Tasks are picked up from tasks directory automatically. See how to create a new task in the `Creating a new task` chapter.
- `-name`: state will be saved in `save/<name>` folder. Necessary to provide if using TB.
- `-restore <checkpoint file>`: restores everything, including the command line arguments, from a checkpoint file. If any other argument is specified, it overwrites the one found in the checkpoint.
- `-reset 1`: do not load checkpoint from `save/<name>` but restart training.
- `-log`: can be `tb` for tensorboard or `wandb` for Weights & Biases. All supported plot types are defined in `framework/visualize/plot.py` and support logging on both. If `tb` is specified, the run will start a Tensorboard session on port 7000 (or the next available)
- `-gpu <index>`: which GPU to use. Leave empty for allocating the next empty one.
- `-lr <learning rate>`: specify learning rate
- `-batch_size <batch size>`: specify batch size
- `-wd`: weight decay
- `-stop_after <n_iters>`: terminate after this many iterations. It also sets the amount of steps for the LR scheduler if used.
- `-amp 1`: use mixed-precision training
- `-grad_clip <max norm>`: clip gradients to the this max norm. 1 by default. Specify `none` to disable.
- `-lr_sched.type cos`: use cos learning rate decay
- `-lr_warmup <n_iters>`: use linear LR warmup for this many steps.
- `-load_pretrained_model <checkpoint file>`: loads the model only, but not the arguments, opitmizer state, etc, from a checkpoint.
- `-length_bucketed_sampling 1`: groups examples of similar length into batches to save compute wasted for padding. Only works for some datasets.
- `-save_interval <n_iters>`: how often to save checkpoints.
- `-test_interval <n_iters>`: how often to run automatic tests.
- `-test_only 1`: run only a validation pass.
- `-per_device_batch_size <batch size>`: specify the per-GPU batch size. Microbatching (gradient accumulation) will be used to ensure that the actual batch size is <= than the specified. Uneven division is supported.
- `-n_microbatch <number of microbatching steps>`: manually specify the number of microbatches. Mutually exclusive with `per_device_batch_size`.

There are many other useful default arguments, defined in `framework/task/task.py`, `framework/task/simple_task.py` and `framework/helpers/training_helper.py`.
