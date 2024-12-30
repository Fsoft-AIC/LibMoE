from .register import register_moe
from .moe import MoeLayer


@register_moe("smoe")
class SMoeLayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)


    def forward(self, x,  return_id_experts = False,  is_vision = False):
        return super().forward(x,  return_id_experts, is_vision)
