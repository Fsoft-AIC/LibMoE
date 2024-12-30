import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F
import copy


class MoeLayer(nn.Module):
    
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__()
        """
        Initializes the Mixture of Experts (MoE) layer.

        Args:
            in_embed_dim (int): Dimension of the input embeddings. Default is 768.
            out_embed_dim (int): Dimension of the output embeddings. Default is 768.
            num_of_experts (int): Number of expert networks in the MoE layer. Default is 4.
            num_selected (int): Number of experts to select per input. Default is 2.
            expert (nn.Module or None): A custom expert module to use. If None, a default expert module will be created.
            gate (nn.Module): A gating network of MoE model
        """
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.num_of_experts = num_of_experts
        self.num_selected = num_selected

        if expert is None:
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(in_embed_dim, out_embed_dim), 
                nn.GELU(), 
                nn.Linear(out_embed_dim, out_embed_dim)) for _ in range(num_of_experts)])
        else:
            if isinstance(expert, nn.ModuleList):
                print("Training from scratch ...")
                self.experts = expert
            else:
                self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])
        
        self.gate = nn.Linear(in_embed_dim, num_of_experts, bias=False)
        self.args = args

    
    def zloss(self, gate_logits, gate_softmax = None):
        """
        Computes the z-loss based on the gating logits.

        The z-loss is a measure of how uniformly the gating logits are distributed. 
        It encourages sparsity in the gating distribution by penalizing the logarithm 
        of the sum of the exponentials of the logits.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            torch.Tensor: The computed z-loss value.
        """
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        return router_z_loss

    def balanceloss(self, selected_experts, gate_softmax):
        """
        Computes the balance loss for the selected experts.

        This loss measures how evenly the gating softmax distribution is distributed
        among the selected experts. It encourages a balanced distribution across experts
        by comparing the density of selected experts with the density of the overall gating softmax.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts 
            gate_softmax (torch.Tensor): Softmax probabilities for each expert 

        Returns:
            torch.Tensor: The computed balance loss value.
        """        
        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_of_experts ** 2)
        return balance_loss


    def topk_expert(self, gate_logits):
        """
        Selects the top-k experts based on the gating logits.

        This method computes the softmax of the gating logits to obtain the probabilities,
        then selects the top-k experts with the highest probabilities for each input sample.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            tuple:
                - weights (torch.Tensor): The softmax probabilities of the top-k experts.
                - selected_experts (torch.Tensor): Indices of the top-k experts.
                - gate_softmax (torch.Tensor): The softmax probabilities for all experts.
        """
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        
        return weights, selected_experts, gate_softmax
    
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs = None):
        """
        Compute the output by routing through the selected experts.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts.
            weights (torch.Tensor): Weights of the selected experts.
            results (torch.Tensor): Tensor to store the results.
            x (torch.Tensor): Input tensor to be processed by the experts.

        Returns:
            torch.Tensor: The computed output from the selected experts.
        """

        infor_experts = {}
        
        for i in range(len(self.experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        # if expert_outputs is not None:
        is_expert = expert_outputs is not None
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
            if is_expert:
                out_exp = expert_outputs[i][batch_idx, token_idx]
            else:
                out_exp = expert(x[batch_idx, token_idx])
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp
        
        return results
    def combine_loss(self, selected_experts, gate_softmax, gate_logits):
        # compute balance loss
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        # compute router_z_loss
        router_z_loss = self.zloss(gate_logits, gate_softmax)

        auxiliary_loss = balance_loss * self.args.balance_loss_coef + \
            router_z_loss * self.args.router_z_loss_coef
        
        return auxiliary_loss, balance_loss
    
    def forward(self, x, return_id_experts = False, is_vision = False):
        # compute output
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x)
        # compute loss
        auxiliary_loss, balance_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        if return_id_experts:
            return output, auxiliary_loss, selected_experts, balance_loss
        else:
            return output, auxiliary_loss, None, balance_loss


