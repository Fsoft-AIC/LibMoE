
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F

from .register import register_moe
from .moe import MoeLayer


@register_moe("xmoe")
class XMOE(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__(in_embed_dim, out_embed_dim, num_of_experts, num_selected, expert, args)
   
        
        expert_embeddings = torch.empty(self.num_of_experts, int(num_of_experts / 2))
        
        # torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )
        self.inp_reduction = torch.nn.Linear(in_embed_dim, int(num_of_experts / 2), bias=False)
        self.temperature = 0.3
        self.bias = None

        self.init_gate_weights()
        self.init_expert_weights()
    def init_gate_weights(self):
        
        """
            Initialize the weights and bias of the gating layer.
            We are make sure that gating of the xmoe same init weight setting with other algorithms 
        """
        init_weight = getattr(self.args, "init_weight", True)

        if init_weight == False:
            print("Not init weight")
            return 
     
        device = self.gate.weight.device if self.gate.weight.device != torch.device('meta') else torch.device('cpu')
        # device = self.gate.weight.device
        gate_generator = torch.Generator(device=device)
        gate_generator.manual_seed(42)
        
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=0.02, generator=gate_generator)
     
        print("Initializing weights and bias of the gating layern successfull.")
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
        # Normalize mat1 across the last dimension (D')
        mat1_normalized = F.normalize(mat1.float(), p=2.0, dim=-1, eps=eps)
        # mat2_normalized = mat2.float()
        # print(f"mat1_normalized dtype: {mat1_normalized.dtype}")
        # print(f"mat2_normalized dtype: {mat2_normalized.dtype}")
        # Compute cosine similarity: (B, N, D') @ (D', E) -> (B, N, E)
        cosine_similarity = torch.matmul(mat1_normalized, mat2.float().transpose(0, 1))
        
        return cosine_similarity.type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
    def competition_policy(self, x):
        """
        Implements the competition policy for expert selection.

        Args:
            x (tensor): Input tensor of shape (B, N, D), where:
                - B: Batch size
                - N: Sequence length
                - D: Input feature dimension

        Returns:
            weights (tensor): Tensor of shape (B, N, num_selected) representing the normalized weights for the selected experts.
            selected_experts (tensor): Tensor of shape (B, N, num_selected) containing the indices of the selected experts.
            affinity_softmax (tensor): Softmax probabilities of the affinity scores, with shape (B, N, num_of_experts).
        """
        B, N, D = x.shape

        # Initialize affinity scores tensor
        affinity_scores = torch.zeros(B, N, self.num_of_experts, device=x.device, dtype=x.dtype)

        # Calculate affinity scores based on the norm of each expert's output
        for i in range(self.num_of_experts):
      
            out_i = self.experts[i](x)
            affinity_scores[:, :, i] = torch.mean(F.softplus(out_i), dim = -1)
   
        # Compute softmax of the affinity scores
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=torch.float)
        # Select top experts based on affinity scores
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        # weights = F.softmax(weights, dim=-1).to(x.dtype)
        return weights, selected_experts, affinity_softmax, affinity_scores
    def forward(self, x, return_id_experts = False, is_vision = False):
        # compute output
        reduced_inp = self.inp_reduction(x)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=-1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
            
        gate_logits = self._cosine(reduced_inp, self.expert_embeddings)
        gate_logits = self._make_finite(gate_logits)
        gate_softmax = F.softmax(gate_logits / self.temperature, dim=-1, dtype=torch.float).to(x.dtype)
        weights, selected_experts = self._keepTopk(gate_softmax)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=False)
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        if self.inp_reduction.weight.requires_grad: 
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach()
            }
        if self.inp_reduction.weight.requires_grad == False and return_id_experts == True:
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            
            self.log_metrics['weights'] = weights
            self.log_metrics['balance_loss'] = balance_loss.item()
            self.log_metrics['router_z_loss'] = router_z_loss.item()
            self.log_metrics['gate_softmax'] = gate_softmax
            self.log_metrics['selected_experts'] = selected_experts
            self.log_metrics['router_magine'] = weights[:, :, 0] - weights[:, :, 1]

        return output, auxiliary_loss, None, infor_aux