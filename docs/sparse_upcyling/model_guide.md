# MoE Model Implementation Guide

This guide explains how to implement a custom Mixture-of-Experts (MoE) layer for training Multimodal Large Language Models (MLLMs) using LibMoE. You'll learn how to create a custom MoE layer by subclassing the `MoeLayer` base class and integrating it into the training pipeline.

---

## Overview

The `MoeLayer` class provides a flexible framework for implementing various MoE architectures. Key features include:

- **Expert routing**: Dynamic selection of top-k experts per token
- **Auxiliary losses**: Z-loss and balance loss for training stability
- **Logging metrics**: Built-in support for analyzing expert behavior during evaluation
- **Flexible initialization**: Support for both scratch training and upcycling from dense models

---

## Setup

First, create a new file for your custom MoE layer implementation:

```bash
touch moe_model/model/moe/<my_model_file_name>.py
```

---

## Base Class Interface

All custom MoE layers must subclass `moe_model.model.moe.MoeLayer`. The base class provides the following key methods that you can override:

### Required Methods to Override

```python
from moe_model.model.moe import MoeLayer

class MyMoELayer(MoeLayer):
    """
    Custom MoE layer implementation.
    
    Args:
        in_embed_dim (int): Input embedding dimension. Default: 768
        out_embed_dim (int): Output embedding dimension. Default: 768
        num_of_experts (int): Number of expert networks. Default: 4
        num_selected (int): Number of experts to select per token. Default: 2
        expert (nn.Module or None): Custom expert module. If None, default MLP experts are created
        args: Training arguments containing loss coefficients and other configs
    """
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        # Add custom initialization here
        
    def zloss(self, gate_logits, gate_softmax=None):
        """
        Compute z-loss to encourage sparse gating distributions.
        
        Args:
            gate_logits (torch.Tensor): Logits from the gating network
            gate_softmax (torch.Tensor, optional): Softmax probabilities
            
        Returns:
            torch.Tensor: Computed z-loss value
        """
        # Override for custom z-loss implementation
        return super().zloss(gate_logits, gate_softmax)
    
    def balanceloss(self, selected_experts, gate_softmax):
        """
        Compute balance loss to encourage even expert utilization.
        
        Args:
            selected_experts (torch.Tensor): Indices of selected experts
            gate_softmax (torch.Tensor): Softmax probabilities for all experts
            
        Returns:
            torch.Tensor: Computed balance loss value
        """
        # Override for custom balance loss implementation
        return super().balanceloss(selected_experts, gate_softmax)
    
    def topk_expert(self, gate_logits):
        """
        Select top-k experts based on gating logits.
        
        Args:
            gate_logits (torch.Tensor): Logits from the gating network
            
        Returns:
            tuple:
                - weights (torch.Tensor): Normalized weights for selected experts
                - selected_experts (torch.Tensor): Indices of selected experts
                - gate_softmax (torch.Tensor): Softmax probabilities for all experts
        """
        # Override for custom expert selection strategy
        return super().topk_expert(gate_logits)
    
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs=None, return_topk_outputs=False):
        """
        Compute MoE output by routing through selected experts.
        
        Args:
            selected_experts (torch.Tensor): Indices of selected experts
            weights (torch.Tensor): Weights for selected experts
            results (torch.Tensor): Tensor to accumulate results
            x (torch.Tensor): Input tensor
            expert_outputs (torch.Tensor, optional): Pre-computed expert outputs
            return_topk_outputs (bool): Whether to return diversity loss
            
        Returns:
            torch.Tensor or tuple: MoE output, optionally with diversity loss
        """
        # Override for custom routing and aggregation
        return super().compute_moe(selected_experts, weights, results, x, expert_outputs, return_topk_outputs)
    
    def forward(self, x, return_id_experts=False):
        """
        Forward pass through the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            return_id_experts (bool): Whether to log expert selection metrics
            
        Returns:
            tuple:
                - output (torch.Tensor): MoE layer output
                - auxiliary_loss (torch.Tensor): Combined auxiliary losses
                - None: Placeholder for compatibility
                - infor_aux (dict): Dictionary containing loss components for logging
        """
        # Implement custom forward logic
        pass
```

---

## Logging Metrics for Analysis

The `MoeLayer` class includes a `log_metrics` dictionary for tracking expert behavior during evaluation. For detailed information on how to analyze expert routing patterns, interpret metrics, and optimize your MoE model, see the [Expert Behavior Analysis Guide](instruction_analysis_moe_vlm.md).

**Quick reference**: Set `return_id_experts=True` during evaluation to collect metrics like:
- Expert selection entropy
- Expert utilization distribution  
- Per-layer routing patterns
- Diversity and specialization metrics

---

## Auxiliary Loss Information (`infor_aux`)

The `forward` method returns an `infor_aux` dictionary containing auxiliary loss components for logging and monitoring during training. This is crucial for tracking MoE training stability.

### Structure

```python
# At the end of forward() method
# Log information for wandb or print to terminal if needed
# You can add more information to the infor_aux dictionary if needed
# But if it impacts gradient calculation, then clone and detach the tensor
infor_aux = {
    "balance_loss": balance_loss.clone().detach(),
    "router_z_loss": router_z_loss.clone().detach()
}

return output, auxiliary_loss, None, infor_aux
```

### Important Guidelines

1. **Always use `.clone().detach()`** when adding tensors to `infor_aux`
   - Prevents unintended gradient flow
   - Avoids memory leaks during logging
   - Ensures logging doesn't affect training

2. **Add custom metrics** as needed for your MoE variant:
   ```python
   infor_aux = {
       "balance_loss": balance_loss.clone().detach(),
       "router_z_loss": router_z_loss.clone().detach(),
       "custom_metric": my_metric.clone().detach(),  # Your custom metric
       "expert_confidence": confidence_score.clone().detach()
   }
   ```

3. **Use for monitoring** during training:
   - Track in wandb/tensorboard
   - Print to terminal for debugging
   - Analyze training stability


---

## Weight Initialization

The base class provides methods for initializing expert and gate weights:

### Expert Weight Initialization

```python
def init_expert_weights(self, std=0.02):
    """
    Initialize weights of all experts with normal distribution.
    
    Args:
        std (float): Standard deviation for initialization. Default: 0.02
    """
    # Controlled by args.init_weight flag
    # Uses seed=42 for reproducibility
```

### Gate Weight Initialization

```python
def init_gate_weights(self, std=0.02):
    """
    Initialize gating network weights with normal distribution.
    
    Args:
        std (float): Standard deviation for initialization. Default: 0.02
    """
    # Controlled by args.init_weight flag
    # Uses seed=42 for reproducibility
```

**Note:** Both methods respect the `args.init_weight` flag. Set it to `False` to skip initialization when loading pretrained weights.

---

## Registration

To integrate your custom MoE layer into LibMoE, you need to register it using the `@register_moe` decorator:

### Step 1: Register Your MoE Layer

In your MoE layer file (`moe_model/model/moe/<my_model_file_name>.py`):

```python
from .register import register_moe
from .moe import MoeLayer

@register_moe("my_moe_model")
class MyMoELayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
        # Custom initialization
    
    def forward(self, x, return_id_experts=False):
        # Custom forward implementation
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        
        auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(
            selected_experts, gate_softmax, gate_logits
        )
        
        # Prepare logging information
        infor_aux = {
            "balance_loss": balance_loss.clone().detach(),
            "router_z_loss": router_z_loss.clone().detach()
        }
        
        return output, auxiliary_loss, None, infor_aux
```

### Step 2: Import in `moe/__init__.py`

Add your MoE layer to `moe_model/model/moe/__init__.py`:

```python
# ... existing imports ...
from .<my_model_file_name> import MyMoELayer  # Add your custom MoE layer
```

### Step 3: Export in `model/__init__.py`

Export your MoE layer in `moe_model/model/__init__.py`:

```python
from .moe import (
    SMoEPlusPlus, 
    SharedExpertV3,
    # ... other MoE layers ...
    MyMoELayer,  # Add your custom MoE layer here
)
```

This ensures your MoE layer is registered and available for use in training scripts.

### What Gets Logged

When `return_id_experts=True`, the evaluation script at ./vision_language_model/scripts/eval/run_eval.sh logs:

- **Entropy scores**: Measure of routing uncertainty
- **Expert distribution**: Which experts are selected most frequently
- **Per-layer metrics**: Routing patterns across different layers
- **Diversity loss**: Expert specialization metrics

These metrics are saved in the evaluation output and can be used to:
- Debug routing issues
- Understand expert specialization
- Optimize hyperparameters (e.g., `num_selected`, loss coefficients)

---

## Summary

To implement a custom MoE layer:

1. ✅ Create a new file in `moe_model/model/moe/`
2. ✅ Subclass `MoeLayer` and override necessary methods
3. ✅ Register with `@register_moe("your_name")`
4. ✅ Import in `moe_model/model/moe/__init__.py`
5. ✅ Export in `moe_model/model/__init__.py`
6. ✅ Use `--moe_name your_name` in training
7. ✅ Analyze with `return_id_experts=true` during evaluation

For more examples, see the existing MoE implementations in the `moe_model/model/moe/` directory.