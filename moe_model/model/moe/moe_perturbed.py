import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from moe_model.model.moe.register import register_moe
import torch.nn.functional as F
 
from .moe import MoeLayer
 
@register_moe("smoe_perturbed")
class MoEPerturbedCosingGating(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None, theta=0.1):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
 
        self.theta = theta
        expert_embeddings = torch.empty(self.num_of_experts, int(num_of_experts / 2))
       
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
       
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )
       
        self.inp_reduction = torch.nn.Linear(in_embed_dim, int(num_of_experts / 2), bias=False)
        self.temperature = 0.3
    def _keepTopk(self, x):
 
        weights, selected_experts  = torch.topk(x, k = self.num_selected, dim=2)
        weights = torch.softmax(weights, dim=2)
        return weights, selected_experts
    def _cosine(self, mat1, mat2, eps=1e-4):
        """
        Compute the cosine similarity between mat1 and mat2.
 
        Args:
            mat1 (torch.Tensor): Input tensor of shape (B, N, D')
            mat2 (torch.Tensor): Expert embeddings of shape (E, D')
            eps (float): Small value to avoid division by zero
 
        Returns:
            torch.Tensor: Cosine similarity scores of shape (B, N, E)
        """
 
        mat1_normalized = mat1.float() / (mat1.norm(p=2, dim=-1, keepdim=True) + self.theta)
 
        # Compute cosine similarity: (B, N, D') @ (D', E) -> (B, N, E)
        cosine_similarity = torch.matmul(mat1_normalized, mat2.float().transpose(0, 1))
       
        return cosine_similarity.type_as(mat1)
 
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
   
    def forward(self, x, return_id_experts = False, is_vision = False):      
        # compute output
        # compute output
        reduced_inp = self.inp_reduction(x)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=-1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / (expert_embeddings_norm  + self.theta))
           
        gate_logits = self._cosine(reduced_inp, self.expert_embeddings)
        gate_logits = self._make_finite(gate_logits)
        gate_softmax = F.softmax(gate_logits / self.temperature, dim=-1, dtype=torch.float).to(x.dtype)
        weights, selected_experts = self._keepTopk(gate_softmax)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
       
        auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
       
        if return_id_experts:
            return output, auxiliary_loss, gate_softmax
        else:
            return output, auxiliary_loss, None, balance_loss
