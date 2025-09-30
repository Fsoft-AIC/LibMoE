# New MoE Model Guide
To properly train a given Multimodal LLM, we require the implementation of a wrapper class that subclasses the `moe_model.model.moe.MoeLayer` class. This wrapper class defines how toolkitmoe should train with your MoE layer. This guide walks you through writing this MoE layer subclass and adding it to the library!
## Setup

First, we'll create a new file for our MoE layer model.

```bash
touch moe_model/model/moe/<my_model_file_name>.py
```
## Interface

All new MoE layers need to subclass `the moe_model.model.moe.MoeLayer` class:


```bash
class MyMoElayer(MoELayer):
    #...
    def gate_modify(self):

        #...

    def zloss(self, gate_logits):

        #...

        return loss
    def balanceloss(self, selected_experts, gate_softmax):

        #...

        return loss
    def topk_expert(self, gate_logits):

        #...

        return weights, selected_experts, gate_softmax

    def compute_moe(self, selected_experts, weights, results, x):

        #...
        return result
```
## Registration

To complete the setup of the new MoE layer, we need to register it using @register_moe from `moe_model.model.moe.register.py`. This decorator registers your MoE layer in the library:

```bash 
from .register import register_moe
from .moe import MoeLayer


@register_moe(my_name_model)
class MyMoElayer(MoELayer):
    #...
```

The second, we need to import model into file moe_model.model.__init__.py, to practice progress registry new moe layer:

```bash
from .<my_model_file_name> import MyMoELayer
```
## Training Script

Finally, update the training script to use your new MoE layer by specifying the moe_name argument:

```bash
deepspeed --include localhost:$ID_GPUS moe_model/train/train_mem.py \
    # ...
    --mm_projector_type moe \
    --mlp_smoe true \
    --clip_smoe true \
    --moe_name my_name_model \
    # ...

```