# MoE Pretraining Configuration Guide

Read this before launching a language-model MoE pretraining run.

This guide explains how to read a sweep YAML, what resources to check before launch, and how the main hyperparameter groups map to the training code. It uses this reference config as the worked example:

`language_modeling/sweeps/154M/slimpajama_moe_no_attmoe_154M_sigmoid_standard_lb.yaml`

## Before You Run

- Confirm the Python environment has the project dependencies, including PyTorch, W&B, SentencePiece, zstandard, Triton, and dataset download access.
- Confirm dataset cache/download access. For SlimPajama configs, first runs may spend time streaming, tokenizing, and caching dataset shards.
- Confirm W&B is configured because this sweep uses `log: wandb`.
- Confirm GPUs `0,1,2,3` are available, or edit the YAML before launching.
- Confirm memory is sufficient for global `batch_size` 64, `per_device_batch_size` 16, and sequence length 1024.
- Confirm `MOE_TYPE` if you want a non-default MoE implementation. If unset, the code uses the default exposed by `language_modeling/layers/__init__.py`.
- Confirm the run name and checkpoint location behavior. If `name` is not set in the YAML, `language_modeling/run.py` derives it from the YAML filename.

## How To Launch

The YAML command block is W&B-sweep style:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```

For a local single run through the repo helper, run from `language_modeling/` so the YAML `program: main.py` resolves correctly:

```bash
cd /mnt/d/workspace/LibMoE/language_modeling
python3 run.py sweeps/154M/slimpajama_moe_no_attmoe_154M_sigmoid_standard_lb.yaml
```

If your launcher already sets the working directory to the repository root, use an equivalent wrapper that resolves `language_modeling/main.py` correctly.

For W&B sweeps, use the selected YAML file with W&B tooling. In the worked example, the sweep objective is to maximize `validation/mean_accuracy`.

## Worked Example: Run Identity

- Task: `slimpajama_transformer`
- Dataset: SlimPajama train and validation splits through `SlimpajamaTransformer`
- Tokenizer: SentencePiece, using the task default vocabulary size of 8000 pieces unless overridden elsewhere
- Model path: `SlimpajamaTransformer -> LMBase -> TransformerLMMixin -> TransformerLanguageModel`
- Model family: pre-layer-norm Transformer with MoE feed-forward blocks
- Sweep method: grid
- Sweep metric: maximize `validation/mean_accuracy`

## Architecture

The worked example builds a 154M-scale pre-layer-norm MoE Transformer. It uses 16 Transformer encoder layers, hidden size 512, 4 attention heads, and a projected attention head size of 82. The configured variant is `preln_moe`, so the sparse MoE is used in the feed-forward path. This YAML does not enable MoE attention.

The universal group size is set to 16. In this exact variant, the model is not using a `*_universal` transformer variant, so this value is present in the config but does not change layer sharing unless the transformer variant is changed to a universal variant.

Dropout and stochastic layer drop are disabled in this run. That makes the run easier to compare against other routing/load-balancing variants, but it also means regularization comes mainly from optimization choices and MoE load balancing rather than dropout.

## MoE Routing And Load Balancing

In the worked example, the feed-forward MoE has 66 experts, each with expert hidden size 128. `pkm.n_heads` is 8, which is passed to the MoE layer as the number of selected experts per token. Routing uses sigmoid selection, meaning expert probabilities are computed independently before top-k selection, rather than using a softmax gate over all experts.

Load balancing uses `standard` perplexity regularization with strength 0.01. The MoE layer adds this through the framework regularization interface, and the training loop aggregates it into the total loss. Expert dropout is disabled, so no experts are randomly masked during routing.

## Training, Data, And Batch Shape

The worked example trains for 100000 optimizer steps. Each training example has 1024 tokens. With global batch size 64, each optimizer step processes approximately 65536 tokens before accounting for distributed partitioning and any implementation-specific padding or sampling details.

The worked example GPU list is `0,1,2,3`. With `per_device_batch_size` 16, the intended layout is four devices with 16 samples per device, matching the global batch size of 64. `n_microbatch` is null, so the framework does not force manual microbatch splitting unless another memory rule triggers it.

The worked example SlimPajama validation ratio is 0.005. In `SlimpajamaTransformer`, this scales the number of validation tokens used for the validation split, keeping validation cheaper than a full validation pass.

## Optimization, Logging, And Checkpoints

The worked example uses AdamW with learning rate 0.00025, weight decay 0.01, gradient clipping at 0.1, and AMP enabled. The learning-rate schedule is cosine annealing with `min_lr_multiplier` 0.1, so the minimum LR is 10 percent of the base LR. Cosine scheduling requires `stop_after` to be set; this config sets it to 100000.

Checkpoints are saved every 10000 steps. Validation also runs every 10000 steps. Detailed MoE and layer diagnostic logging is configured every 500 steps where supported by the model/layers.

Full LM evaluation benchmark datasets are disabled in the worked example with `lm.eval.enabled: 0`. The normal SlimPajama validation set still exists and is used for pretraining validation.

## Complete Hyperparameter Reference

Each YAML parameter from the worked example appears exactly once below, grouped by the part of the run it controls.

### Run, Logging, And Checkpointing

| YAML key | Value | Meaning | Code/source area | Change guidance |
|---|---:|---|---|---|
| `log` | `wandb` | Enables W&B logging instead of TensorBoard-only logging. | `framework/helpers/training_helper.py` | Use `tb` for local TensorBoard-only runs or `all` for both, if supported. |
| `task` | `slimpajama_transformer` | Selects the SlimPajama language-modeling task. | `tasks/slimpajama_transformer.py` | Change only when switching dataset/task implementation. |
| `test_interval` | `10000` | Runs validation every 10000 training steps. | `framework/task/task.py` | Lower for more frequent validation, higher to reduce validation cost. |
| `save_interval` | `10000` | Saves checkpoints every 10000 steps. | `framework/helpers/training_helper.py` | Lower for safer recovery; higher to reduce storage and checkpoint overhead. |
| `details_log_interval` | `500` | Interval for detailed layer/MoE diagnostics where supported. | `tasks/transformer_lm_mixin.py`, MoE layers | Lower gives more diagnostics but more logging overhead. |
| `lm.eval.enabled` | `0` | Disables full LM benchmark evaluation datasets during pretrain. | `tasks/lm_eval_mixin.py` | Enable only when you want benchmark eval during training. |

### Model Architecture

| YAML key | Value | Meaning | Code/source area | Change guidance |
|---|---:|---|---|---|
| `state_size` | `512` | Transformer hidden dimension and main model width. | `main.py`, `tasks/transformer_lm_mixin.py` | Increasing this raises memory, compute, and parameter count. |
| `transformer.encoder_n_layers` | `16` | Number of Transformer encoder layers. | `main.py`, `tasks/transformer_lm_mixin.py` | Must be compatible with universal grouping if switching to a universal variant. |
| `transformer.n_heads` | `4` | Number of attention heads in each Transformer layer. | `main.py`, `tasks/transformer_lm_mixin.py` | Keep compatible with hidden size and projection settings. |
| `transformer.variant` | `preln_moe` | Uses pre-layer-norm Transformer layers with MoE feed-forward blocks. | `tasks/transformer_lm_mixin.py` | Changing this can switch architecture family. |
| `transformer.head_projection_size` | `82` | Projected attention head dimension used by the Transformer layer. | `tasks/transformer_lm_mixin.py` | Tune with attention shape and model-size targets. |
| `transformer.universal.group_size` | `16` | Layer sharing group size for universal variants. | `tasks/transformer_lm_mixin.py` | Has no practical effect unless using a `*_universal` variant. |
| `transformer.p_drop_layer` | `0.0` | Probability of dropping a layer update during training. | `models/transformer_language_model.py` | Leave at 0.0 unless testing stochastic depth. |
| `dropout` | `0.0` | Main Transformer dropout probability. | `main.py`, `models/transformer_language_model.py` | Increase for regularization; leave at 0.0 for controlled comparisons. |
| `lm.trafo.context_blocks` | `0` | Number of previous context blocks carried in Transformer state. | `tasks/transformer_lm_mixin.py`, `models/transformer_language_model.py` | Increase only when training with recurrent/context memory. |

### MoE Routing And Load Balancing

| YAML key | Value | Meaning | Code/source area | Change guidance |
|---|---:|---|---|---|
| `moe.n_experts` | `66` | Number of feed-forward MoE experts. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | More experts increase sparse capacity and routing cost. |
| `moe.expert_size` | `128` | Hidden size inside each expert. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | Larger experts increase expert capacity and parameter count. |
| `pkm.n_heads` | `8` | Top-k expert selections per token for the MoE layer. | `tasks/transformer_lm_mixin.py` | Higher values activate more experts per token and increase compute. |
| `moe.selection_mode` | `sigmoid` | Uses independent sigmoid routing scores before expert top-k. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | Use `gate` for softmax-style routing. |
| `moe.perplexity_reg_mode` | `standard` | Selects the standard load-balancing/perplexity regularization mode. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | Change only when comparing balancing objectives. |
| `moe.perplexity_reg` | `0.01` | Strength of MoE load-balancing regularization. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | Lower weakens balancing; higher can over-constrain routing. |
| `moe.drop_expert` | `0.0` | Probability of dropping experts during routing. | `tasks/transformer_lm_mixin.py`, `layers/moe_layer.py` | Increase only if you want expert-level routing regularization. |

### Optimization And Schedule

| YAML key | Value | Meaning | Code/source area | Change guidance |
|---|---:|---|---|---|
| `optimizer` | `adamw` | Uses AdamW optimizer. | `framework/task/simple_task.py` | Other supported values include `adam`, `sgd`, and `adagrad`. |
| `lr` | `0.00025` | Base optimizer learning rate. | `framework/task/task.py` | Tune with batch size, schedule, and stability. |
| `wd` | `0.01` | Optimizer weight decay. | `framework/task/task.py`, `framework/task/simple_task.py` | Tune with optimizer and model size. |
| `lr_sched.type` | `cos` | Uses cosine annealing learning-rate schedule. | `framework/task/task.py` | Requires `stop_after`; alternatives include `step` and `noam`. |
| `min_lr_multiplier` | `0.1` | Minimum cosine LR as a fraction of base LR. | `framework/task/task.py` | With LR 0.00025, final minimum is 0.000025. |
| `grad_clip` | `0.1` | Clips gradient norm before optimizer step. | `framework/task/task.py`, `framework/task/simple_task.py` | Raise if clipping is too aggressive; lower if training is unstable. |
| `amp` | `1` | Enables automatic mixed precision when CUDA is available. | `framework/task/simple_task.py` | Disable for debugging numerical issues, at higher memory cost. |
| `stop_after` | `100000` | Stops training after 100000 iterations. | `framework/task/simple_task.py` | Required for cosine scheduling in this framework. |

### Data, Sequence Length, And Batch Layout

| YAML key | Value | Meaning | Code/source area | Change guidance |
|---|---:|---|---|---|
| `lm.unroll` | `1024` | Sequence length in tokens for training samples. | `tasks/lm_base.py`, `tasks/slimpajama_transformer.py` | Larger values improve context length but increase memory. |
| `lmds.valid_ratio` | `0.005` | Fraction-like scale for SlimPajama validation token budget. | `tasks/slimpajama_transformer.py` | Increase for more thorough validation; decrease for faster validation. |
| `batch_size` | `64` | Global training batch size. | `framework/task/task.py`, `framework/helpers/training_helper.py` | Coordinate with GPU count and per-device memory. |
| `per_device_batch_size` | `16` | Intended per-device batch size. | `framework/task/task.py` | Keep aligned with global batch and number of GPUs. |
| `n_microbatch` | `null` | Does not force a fixed number of manual microbatches. | `framework/task/task.py`, `framework/task/simple_task.py` | Set when you need explicit gradient accumulation chunks. |
| `gpu` | `0,1,2,3` | Selects GPUs visible/used by the run helper. | `framework/helpers/training_helper.py` | Edit to match the machine before launching. |

## Expected Outputs

- W&B run logs under the `lm` project.
- Checkpoints under the run save/checkpoint directory every 10000 steps.
- Startup args and logs under the run save directory.
- SlimPajama validation metrics every 10000 steps.

## Quick Sanity Checks

- The launch command should print the resolved `python3 main.py ...` command.
- The startup log should show `task=slimpajama_transformer`.
- The model parameter count should print after model creation.
- The first dataset run may be slow while cache artifacts are created.
- Validation metrics should appear at step 10000 if the run reaches that step.
