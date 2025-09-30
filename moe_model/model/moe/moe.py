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
   
        self.aux_loss = {
            "zloss": self.zloss,
            "balanceloss": self.balanceloss
            
        }
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
        self.is_vision = False
        self.log_metrics = {}
    

    def init_expert_weights(self, std=0.02):
        """
        Initialize the weights of all experts in the MoE layer.
        
        Args:
            std (float): Standard deviation for normal initialization. Default is 0.02.
        """
        init_weight = getattr(self.args, "init_weight", True)
        
        if not init_weight:
            print("Not initializing expert weights")
            return
        
        # Determine device for initialization
        device = next(self.experts[0].parameters()).device if next(self.experts[0].parameters()).device != torch.device('meta') else torch.device('cpu')
        expert_generator = torch.Generator(device=device)
        expert_generator.manual_seed(42)  # Same seed as gate for consistency
        
        for expert in self.experts:
            for name, param in expert.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=std, generator=expert_generator)
                elif 'bias' in name and param is not None:
                    nn.init.constant_(param, 0.0)
        
        print(f"Initialized weights of experts successfully with device: {device}")
    def init_gate_weights(self, std = 0.02):
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
        """
        Initialize the weights and bias of the gating layer.
        """
        nn.init.normal_(self.gate.weight, mean=0.0, std=std, generator=gate_generator)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0.0)
        print(f"Initializing weights and bias of the gating layer succefull with device: {device}")
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


    

    def experts_diversity_loss(self, expert_outputs):
        """
        expert_outputs: Tensor shape [B, N, K, D]
            - B: batch size
            - N: sequence length
            - K: number of selected experts
            - D: dimension of each expert output

        Mục tiêu: phạt khi các expert outputs 'quá giống nhau'.
        Ta sẽ tính độ tương đồng cos trung bình giữa mọi cặp (i, j) trong K experts, rồi lấy mean.
        """
        expert_outputs = expert_outputs.to(torch.float32)
        B, N, K, D = expert_outputs.shape

        # Bước 1: Chuẩn hoá (L2-normalize) theo chiều D để tính Cosine Similarity
        # Shape sau chuẩn hoá vẫn là [B, N, K, D]
        normalized = F.normalize(expert_outputs, p=2, dim=-1)

        # Bước 2: Đưa (B, N) về 1 batch lớn để dễ tính bmm
        # Ta reshape thành [B*N, K, D]
        normalized_reshape = normalized.view(B*N, K, D)  # => [B*N, K, D]

        # Bước 3: Tính ma trận similarity bằng bmm:
        # [B*N, K, D] x [B*N, D, K] -> [B*N, K, K]
        similarity_matrix = torch.bmm(
            normalized_reshape, 
            normalized_reshape.transpose(1, 2)
        )  # => [B*N, K, K]

        # Bước 4: Loại bỏ độ tương đồng với chính nó (đường chéo)
        # identity = [K, K], shape broadcast được cho [B*N, K, K]
        mask = 1 - torch.eye(K, device=expert_outputs.device)
        
        similarity_matrix = similarity_matrix * mask
        
        similarity_matrix = F.relu(similarity_matrix)

        # Bước 5: Tính trung bình trên tất cả các batch, token, và cặp expert
        # similarity_matrix có shape [B*N, K, K]. Số phần tử hợp lệ = B*N * K * (K-1)
        loss = similarity_matrix.mean()

        return loss
    
    def compute_moe(self, selected_experts, weights, results, x, expert_outputs = None, return_topk_outputs = False):
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
        B, N, D = x.shape

        for i in range(len(self.experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        if return_topk_outputs == True:
            expert_outputs_topk = torch.zeros(x.shape[0], x.shape[1], self.num_of_experts, self.out_embed_dim, device=x.device, dtype=x.dtype)
        
        # if expert_outputs is not None:
        is_expert = expert_outputs is not None
        for i in range(self.num_of_experts):

            expert = self.experts[i]
       
            batch_idx, token_idx, topk_idx = infor_experts[i]
            if batch_idx.numel() == 0 : continue
            if is_expert:
                out_exp = expert_outputs[i][batch_idx, token_idx]
            else:
                out_exp = expert(x[batch_idx, token_idx])
            if return_topk_outputs == True:
                expert_outputs_topk[batch_idx, token_idx, i] = out_exp
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp
        if return_topk_outputs:
            idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, selected_experts.shape[-1], x.size(-1))
            topk_expert_outputs = torch.gather(expert_outputs_topk, dim=2, index=idx_expanded)
            diver_loss = self.experts_diversity_loss(topk_expert_outputs)
            if x.requires_grad == False: 
                self.log_metrics['diver_loss'] = diver_loss.item()
            else:
                return results, diver_loss
        return results
    
    
    
    def combine_loss(self, selected_experts, gate_softmax, gate_logits, acitve_zloss =True):
        # compute balance loss
        balance_loss = self.balanceloss(selected_experts=selected_experts, gate_softmax=gate_softmax)
        
        router_z_loss = torch.tensor(0.0, device=selected_experts.device, dtype=selected_experts.dtype)
        
        if acitve_zloss:
            # compute router_z_loss
            router_z_loss = self.zloss(gate_logits, gate_softmax)

            auxiliary_loss = balance_loss * self.args.balance_loss_coef + \
                router_z_loss * self.args.router_z_loss_coef
            
        else: 
            
            auxiliary_loss = balance_loss * self.args.balance_loss_coef 
            
        return auxiliary_loss, balance_loss, router_z_loss
    def compute_entropy_score(self, probabilities, normalize=False, eps=1e-10):
        """
        Compute the Shannon entropy score of a probability distribution using PyTorch.
        
        Args:
            probabilities (torch.Tensor): Tensor of probabilities or unnormalized scores.
                                        Shape: (N,) for single distribution or (B, N) for batch.
            normalize (bool): If True, apply softmax to convert scores to probabilities.
            eps (float): Small value to avoid log(0).
        
        Returns:
            torch.Tensor: Entropy score in bits (scalar for single dist, shape (B,) for batch).
            
        Raises:
            ValueError: If probabilities are invalid (e.g., negative, empty, or sum to zero).
        """
        # Ensure input is a tensor
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities, dtype=torch.float)
        
        # Check for valid input
        if probabilities.numel() == 0:
            raise ValueError("Probability tensor cannot be empty.")
        if torch.any(probabilities < 0):
            raise ValueError("Probabilities cannot be negative.")
        
        probs = probabilities.to(dtype=torch.float32)
    
        if normalize:
            # Chuẩn hóa bằng tổng (thay vì softmax)
            sum_probs = torch.sum(probs, dim=-1, keepdim=True)
            # if sum_probs
            if torch.any(sum_probs == 0):
                return 0
                raise ValueError("Không thể chuẩn hóa: tổng bằng 0.")
            probs = probs / sum_probs
        
        # Giới hạn xác suất trong khoảng [eps, 1.0]
        probs = torch.clamp(probs, min=eps, max=1.0)
        
        # Tính entropy: -sum(p * log2(p))
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
        
        return entropy.to(probabilities.dtype)
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
        # breakpoint()
        # weights, selected_experts = torch.topk(gate_softmax, self.num_selected + 2 )
        
        # weights = weights[:, :, : self.num_selected]
        # selected_experts = selected_experts[:, :, -self.num_selected:]
        
        return weights, selected_experts, gate_softmax
    def forward(self, x, return_id_experts = False):
        # compute output
        gate_logits = self.gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x, return_topk_outputs=True)
        # compute loss
        auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
        
        infor_aux = {
            "balance_loss": balance_loss.clone().detach(),
            "router_z_loss": router_z_loss.clone.detach()
        }
    
        return output, auxiliary_loss, None, infor_aux


