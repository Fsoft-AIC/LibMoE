# 📊 Expert Behavior Analysis Guide for MoE VLM

This guide explains how to analyze and understand the behavior of Mixture-of-Experts (MoE) layers in Vision-Language Models using LibMoE's built-in logging and analysis tools.

---

## Overview

LibMoE provides comprehensive tools for analyzing expert routing behavior during evaluation. The complete analysis toolkit is available in [`analyst_tool`](../../vision_language_model/evaluate/analysis/libmoev2). 

To use the analysis tools, you must first collect and prepare log files in JSON format. This guide walks you through the complete process:

1. **Define metrics** to collect in your MoE layer
2. **Configure evaluation** to log expert behavior
3. **Run evaluation** to generate log files
4. **Analyze results** using the provided toolkit

---

## Step-by-Step Guide

The main evaluation script is located at [`run_eval.sh`](../../vision_language_model/scripts/eval/run_eval.sh). You need to:

1. Define which metrics to collect in [`moe.py`](../../vision_language_model/moe_model/model/moe/moe.py)
2. Enable metric collection in the evaluation script
3. Process the generated logs using the analysis toolkit





### Step 1: Define Metrics to Collect

In your MoE layer's `forward` method ([`moe.py`](../../vision_language_model/moe_model/model/moe/moe.py)), populate the `log_metrics` dictionary when `return_id_experts=True`:

```python
def forward(self, x, return_id_experts=False):
    # ... MoE forward pass ...
    
    # Collect metrics for analysis
    if return_id_experts:
        self.log_metrics["selected_experts"] = selected_experts.clone().detach()
        self.log_metrics["gate_softmax"] = gate_softmax.clone().detach()
        self.log_metrics["gate_logits"] = gate_logits.clone().detach()
        self.log_metrics["weights"] = weights.clone().detach()
        # Add any custom metrics here
    
    return output, auxiliary_loss, None, infor_aux
```

**Important**: Always use `.clone().detach()` to avoid gradient computation issues.

---

### Step 2: Configure Metric Collection

The evaluation script ([`llava.py`](../../vision_language_model/evaluate/lmms_eval/models/llava.py)) automatically processes `log_metrics` from each MoE layer when `return_id_experts=True`.

The script collects and computes the following metrics:

```python
if kwargs['return_id_experts'] == True and image_tensor is not None:
    # Iterate through all MoE layers in the model
    for name, layer in self.model.named_modules():
        if hasattr(layer, 'log_metrics'):
            # Extract layer ID (e.g., "layers.15" -> "15")
            match = re.search(pattern, name)
            if match:
                id_layer = str(match.group(1))
            else:
                id_layer = 'mm_projector'  # Vision projector layer

            log_layers[id_layer] = {}

            # Compute entropy of top-k expert weights
            if 'weights' in layer.log_metrics:
                entropy_weight_topk = self.compute_entropy_topk(layer.log_metrics['weights'])
                log_layers[id_layer]["entropy_weight_topk"] = entropy_weight_topk.item()

            # Compute entropy of all expert probabilities
            if 'gate_softmax' in layer.log_metrics:
                entropy_weight_all = self.compute_entropy_topk(layer.log_metrics['gate_softmax'])
                _, selected_experts_top1 = torch.topk(layer.log_metrics['gate_softmax'], k=1, dim=2)
                num_experts = layer.log_metrics['gate_softmax'].shape[-1]
                log_layers[id_layer]["entropy_weight_all"] = entropy_weight_all.item()
                
                # Compute top-1 expert distribution
                dist_experts_top1 = self.compute_expert_distribution(selected_experts_top1, num_experts)
                log_layers[id_layer]["dist_experts_top1"] = dist_experts_top1.squeeze().tolist()
            
            # Compute top-k expert distribution
            if 'selected_experts' in layer.log_metrics:
                dist_experts_top2 = self.compute_expert_distribution(layer.log_metrics['selected_experts'], num_experts)
                log_layers[id_layer]["dist_experts_top2"] = dist_experts_top2.squeeze().tolist()
            
            # Custom metrics (if defined)
            if 'router_magine' in layer.log_metrics:
                log_layers[id_layer]["router_magine"] = layer.log_metrics['router_magine'][0].tolist()
            
            # Auxiliary loss metrics
            for key in ['balance_loss', 'router_z_loss', 'diver_loss']:
                if key in layer.log_metrics:
                    log_layers[id_layer][key] = layer.log_metrics[key]
            
            # Any other scalar metrics
            for metric, value in layer.log_metrics.items():
                if metric not in log_layers[id_layer] and isinstance(value, (int, float)):
                    log_layers[id_layer][metric] = value
```

**You can customize this section** to add your own metric computations.

---

### Step 3: Run Evaluation with Logging Enabled

Edit [`run_eval.sh`](../../vision_language_model/scripts/eval/run_eval.sh) and enable `return_id_experts=True`:

```bash
task_name=$(echo $task | tr ',' '_')
python3 -m accelerate.commands.launch \
    --main_process_port=29511 \
    --num_processes=$num_gpus \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$PRETRAINED_PATH",conv_template="$part",return_id_experts=true \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_${task_name} \
    --output_path $PRETRAINED_PATH/logs/
```

**Recommended**: Create a dedicated analysis folder for better organization:

```bash
--output_path $PRETRAINED_PATH/analyst/expert_analysis/
```

---

## Output Format

### Directory Structure

After running evaluation, the output directory will contain:

```
0717_2340_llava_mstar_llava_model_args_4596b5/
├── mmstar.json          # Contains expert analysis metrics
├── rank0_metric_eval_done.txt
├── rank1_metric_eval_done.txt
└── results.json
```

### Log File Structure

The `mmstar.json` file contains detailed metrics in the following format:

```json
{
    "logs": [
        {
            "logs_metrics_mlp": [
                {
                    "balance_loss": 1.0177,
                    "router_z_loss": 0.2212
                }
            ],
            "logs_metrics_vision": [
                {
                    "time_inference": 4.2116758823394775,
                    "mm_projector": {
                        "entropy_weight_topk": 0.9833807909281921,
                        "entropy_weight_all": 1.4860320091247559,
                        "dist_experts_top1": [53, 42, 29, 51, 44, 37],
                        "dist_experts_top2": [158, 436]
                    },
                    "0": {
                        "entropy_weight_topk": 0.95,
                        "entropy_weight_all": 1.85,
                        "dist_experts_top1": [15, 25, 35, 25],
                        "dist_experts_top2": [30, 35, 20, 15]
                    }
                }
            ]
        }
    ]
}
```

### Metric Descriptions

| Metric | Description |
|--------|-------------|
| `entropy_weight_topk` | Shannon entropy of top-k selected expert weights |
| `entropy_weight_all` | Shannon entropy of all expert probabilities |
| `dist_experts_top1` | Distribution showing which expert was selected most (top-1) |
| `dist_experts_top2` | Distribution of top-k expert selections |
| `balance_loss` | Load balancing loss value |
| `router_z_loss` | Router z-loss value |
| `time_inference` | Inference time in seconds |

---

## Using the Analysis Toolkit

After collecting the log files, use the provided analysis toolkit to generate insights and visualizations.

### Analysis Tools Location

The complete analysis toolkit is available at:
- **Main toolkit**: [`vision_language_model/evaluate/analysis/`](../../vision_language_model/evaluate/analysis/)
- **LibMoE v2 tools**: [`vision_language_model/evaluate/analysis/libmoev2/`](../../vision_language_model/evaluate/analysis/libmoev2/)
