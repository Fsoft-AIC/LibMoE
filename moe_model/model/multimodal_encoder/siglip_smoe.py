import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Sequence, List, Tuple, Union
from einops import rearrange, repeat, reduce, pack, unpack

from transformers.activations import ACT2FN
from transformers import SiglipVisionConfig

from moe_model.model.moe.register import get_moe
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states
    ):
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # breakpoint()
        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # avg_attn_weights = attn_weights.mean(dim=1)
        # avg_attn_scores = avg_attn_weights.mean(dim=-1) 

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
    
    
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipEncoderMoELayer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        print("Using Siglip Encoder MoE Layer")
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        if args.sparse_upcycling:
            self.moelayer = get_moe(config.moe_name)(
                in_embed_dim = self.embed_dim, 
                out_embed_dim = self.embed_dim, 
                num_of_experts = config.num_experts,
                num_selected = config.num_selected,
                expert = SiglipMLP(config),
                args = args
            )
        else:
            self.moelayer = get_moe(config.moe_name)(
                in_embed_dim = self.embed_dim, 
                out_embed_dim = self.embed_dim, 
                num_of_experts = config.num_experts,
                num_selected = config.num_selected,
                expert = nn.ModuleList([SiglipMLP(config) for _ in range(config.num_experts)]),
                args = args
            )
        
    def forward(
        self,
        hidden_states, 
        return_id_experts = False
    ):
        
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # compute MoE layer result
        results, auxiliary_loss, id_experts, balance_loss = self.moelayer(hidden_states, return_id_experts, is_vision=True)

        hidden_states = residual + results

        outputs = (hidden_states, auxiliary_loss, id_experts, balance_loss)
        return outputs
    
class SiglipEncoder(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        config.moe_name = args.moe_name
        config.num_experts = args.num_experts
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderMoELayer(config, args) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        return_id_experts = False
    ):
        encoder_states = () 
        hidden_states = inputs_embeds
        auxiliary_losses = []
        balance_losses = []
        z_losses = []
        stored_history_experts = {}
        
        if return_id_experts:

            for idx, encoder_layer in enumerate(self.layers):
                encoder_states = encoder_states + (hidden_states,)
                layer_outputs = encoder_layer(hidden_states, return_id_experts)
                hidden_states = layer_outputs[0]
                auxiliary_loss = layer_outputs[1]
                auxiliary_losses.append(auxiliary_loss)
                stored_history_experts[idx] = layer_outputs[2]
        else:
            for idx, encoder_layer in enumerate(self.layers):
                encoder_states = encoder_states + (hidden_states,)
                layer_outputs = encoder_layer(hidden_states = hidden_states)
                hidden_states = layer_outputs[0]
                auxiliary_loss = layer_outputs[1]
                auxiliary_losses.append(auxiliary_loss)
                balance_loss = layer_outputs[3]
                balance_losses.append(balance_loss)
        
        return encoder_states, auxiliary_losses, stored_history_experts ,balance_losses
    
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
    
@dataclass
class SiglipOutput:
    encoder_output: torch.Tensor
    auxiliary_loss: torch.Tensor
    stored_history_experts: Optional[Any]
    balance_loss: torch.Tensor
    extra: Dict[str, Any] = field(default_factory=dict)  
    
class SiglipSMoEVisionTransformer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        config.num_of_experts = args.num_experts
        config.num_selected = args.num_selected

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config, args)

    def forward(
        self,
        pixel_values,
        return_id_experts = False
    ):        
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs, auxiliary_losses, stored_history_experts, balance_losses = self.encoder(inputs_embeds = hidden_states, return_id_experts = return_id_experts)
        return encoder_outputs[-1], torch.stack(auxiliary_losses).mean(), stored_history_experts, torch.stack(balance_losses).mean()

