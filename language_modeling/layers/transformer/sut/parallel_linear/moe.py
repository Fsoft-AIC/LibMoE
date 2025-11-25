# Modified from David Rau's implementation, preamble follows:
#  Sparsely-Gated Mixture-of-Experts Layers.
#  See "Outrageously Large Neural Networks"
#  https://arxiv.org/abs/1701.06538
#
#  Author: David Rau
#
#  The code is based on the TensorFlow implementation:
#  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.layers import RegularizedLayer

from .parallel_experts import ParallelExperts

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    # batch_index = index_sorted_experts // k
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


def logsumexp(x, dim=-1):
    k = torch.max(x, dim=dim, keepdims=True)[0]
    exp_x = torch.exp(x - k)
    sumexp_x = torch.sum(exp_x, dim=dim)
    logsumexp_x = torch.log(sumexp_x) + k.squeeze(dim)
    return logsumexp_x



class MoE(RegularizedLayer, nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0, switchloss=0, zloss=0, miloss=0,
                 bias=False, gating_dropout=0,
                 activation=None, noisy_gating=False,
                 acc_aux_loss=False,
                 mlp_router=True,
                 hidden_size=None):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        if hidden_size is None:
            hidden_size = head_size
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.miloss = miloss
        self.activation = activation

        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if mlp_router:
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Tanh(),
                nn.Dropout(gating_dropout),
                nn.Linear(input_size,
                          2 * num_experts if noisy_gating else num_experts,
                          bias=False)
            )
            out_weight = self.w_gate[-1].weight
        else:
            self.w_gate = nn.Linear(
                input_size,
                2 * num_experts if noisy_gating else num_experts,
                bias=False
            )
            out_weight = self.w_gate.weight

        if self.noisy_gating:
            nn.init.zeros_(out_weight)

    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_count = 0

        self.probs = []
        self.log_probs = []
        self.masks = []

    def update_aux_statistics(self, logits, probs, gates, skip_mask=None):
        lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
        log_probs = torch.log_softmax(logits, dim=-1)

        self.acc_count = self.acc_count + logits.size(0)
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()

        self.probs.append(probs)
        self.log_probs.append(log_probs)
        self.masks.append(skip_mask)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss =  self.num_experts * (
            F.normalize(self.acc_probs, p=1, dim=0) *
            F.normalize(self.acc_freq, p=1, dim=0)
        ).sum()

        zloss = self.acc_lsesq / self.acc_count


        prob = torch.cat(self.probs, dim=0)
        log_prob = torch.cat(self.log_probs, dim=0)

        has_mask = any(m is not None for m in self.masks)

        if has_mask:
            mall = torch.ones((self.probs[0].shape[0],1), dtype=prob.dtype, device=prob.device)
            mask = torch.cat([m if m is not None else mall for m in self.masks], dim=0)

        if has_mask:
            p_x = mask / (mask.sum() + 1e-12)
        else:
            p_x = 1 / prob.nelement()

        p_e = (p_x * prob).sum(0)
        H_e = (p_e * p_e.log()).sum()
        neg_H_e_given_x = (p_x * prob * log_prob).sum()
        miloss = -(neg_H_e_given_x + H_e)

        losses = {
            "cvloss": self.cvloss * cvloss,
            "switchloss": self.switchloss * switchloss,
            "zloss": self.zloss * zloss,
            "miloss": self.miloss * miloss,
        }
        self.init_aux_statistics()
        return losses

    def get_reg_loss(self) -> Dict[str, torch.Tensor]:
        losses = super().get_reg_loss()
        losses.update(self.get_aux_loss_and_clear())
        return losses

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def compute_miloss(self, logits, prob, mask):
        log_prob = torch.log_softmax(logits, dim=-1)
        if mask is None:
            p_x = 1 / logits.shape[0]
        else:
            p_x = mask / (mask.sum() + 1e-12)
        p_e = (p_x * prob).sum(0)
        H_e = (p_e * p_e.log()).sum()
        neg_H_e_given_x = (p_x * prob * log_prob).sum()
        miloss = -(neg_H_e_given_x + H_e)

        return miloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0,
                     noise_epsilon=1e-2, return_indices=False):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.w_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            if self.training:
                noise_stddev = torch.sigmoid(raw_noise_stddev) + noise_epsilon
                eps = torch.randn_like(clean_logits)
                noisy_logits = clean_logits + eps * noise_stddev
                logits = noisy_logits
            else:
                logits = clean_logits
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, (skip_mask == 0), 0)
            logits = torch.masked_fill(logits, (skip_mask == 0), 0)

        if self.training and (sample_topk > 0):
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates
        self.gates = gates.detach()

        loss = 0.
        if self.acc_aux_loss and self.training:
            self.update_aux_statistics(logits, probs, gates, skip_mask)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
            loss += self.miloss * self.compute_miloss(logits, probs, skip_mask)
        if return_indices:
            return loss, top_k_indices
        else:
            return loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
            skip_mask = skip_mask.flatten()[:, None]
        x = x.reshape(-1, emb_size)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, self.input_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y, loss


    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask,
                                 sample_topk=sample_topk,
                                 return_indices=return_indices)
        if return_indices:
            loss, idxs = loss
            idxs = idxs.view(bsz, length, -1)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        if return_indices:
            return y, idxs, loss
        else:
            return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y
