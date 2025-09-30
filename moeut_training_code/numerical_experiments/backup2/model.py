import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import os
import time

class MoEModel:
    def __init__(self, num_experts: int, K: int, num_shared: int=0, router_type="softmax"):
        """
        Initialize a Mixture of Experts model.

        Parameters
        ----------
        num_experts : int
            Number of experts.
        K : int
            Number of experts to keep in the gating function.
        num_shared : int
            Number of shared experts
        router_type : str
            Type of router to use, either 'softmax' or 'sigmoid'.
        """
        self.num_experts = num_experts
        self.K = K
        self.num_shared = num_shared
        self.router_type = router_type

        # Initialize parameters
        self.beta0 = None
        self.beta1 = None
        self.a = None
        self.b = None
        self.sigma = None

        # Initialize shared parameters
        if self.num_shared > 0:
            self.num_experts += self.num_shared
            self.K += self.num_shared

    def init_params(self, true_params, n_samples):
        """
        Starting values for EM algorithm based on true parameters.

        Parameters
        ----------
        true_params : tuple
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true)
        n_samples : int
            Number of samples in the dataset

        Returns
        -------
        tuple
            Tuple of (beta1_init, beta0_init, a_init, b_init, sigma_init)
        """
        n_components = self.num_experts - self.num_shared
        n_features = 1

        (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true) = true_params

        beta1_init = np.zeros((n_components, n_features))
        beta0_init = np.zeros(n_components)
        a_init = np.zeros((n_components, n_features))
        b_init = np.zeros(n_components)
        sigma_init = np.zeros(n_components)
        
        # add a smaller noise
        noise_scale = 0.000005 * n_samples**(-0.083 * 2)
        sigma_noise_scale = 0.000005 * n_samples**(-0.25 * 2)

        for i in range(n_components):
            i_true = np.random.randint(0, beta0_true.size) if i > beta0_true.size - 1 else i

            beta1_init[i] = beta1_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            beta0_init[i] = beta0_true[i_true] + np.random.normal(0, noise_scale)
            a_init[i] = a_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            b_init[i] = b_true[i_true] + np.random.normal(0, noise_scale)
            sigma_init[i] = sigma_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))

        if self.num_shared > 0:
            n_shared_component = self.num_shared
            n_shared_features = 1

            a_shared_init = np.zeros((n_shared_component, n_shared_features))
            b_shared_init = np.zeros(n_shared_component)
            sigma_shared_init = np.zeros(n_shared_component)

            for i in range(n_shared_component):
                i_true = np.random.randint(0, a_shared_true.size) if i > a_shared_true.size - 1 else i
                a_shared_init[i] = a_shared_true[i_true] + np.random.normal(0, noise_scale, size=(n_shared_features))
                b_shared_init[i] = b_shared_true[i_true] + np.random.normal(0, noise_scale)
                sigma_shared_init[i] = sigma_shared_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))
                
            # merge the shared parameters with the expert parameters
            a_init = np.concatenate([a_init, a_shared_init])
            b_init = np.concatenate([b_init, b_shared_init])
            sigma_init = np.concatenate([sigma_init, sigma_shared_init])

        return beta1_init.flatten(), beta0_init, a_init.flatten(), b_init, sigma_init


    def initialize_parameters(self, X, Y, true_params=None):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)
        true_params : tuple, optional
            If provided, initialize parameters based on true parameters.
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true)
        """
        n = len(X)

        if true_params is not None:
            # Initialize parameters based on true parameters
            beta1_init, beta0_init, a_init, b_init, sigma_init = self.init_params(true_params, n)
            self.beta0 = beta0_init
            self.beta1 = beta1_init
            self.a = a_init
            self.b = b_init
            self.sigma = sigma_init
        else:
            # Initialize with random values
            rng = default_rng()
            self.beta0 = rng.normal(size=self.num_experts - self.num_shared)
            self.beta1 = rng.normal(size=self.num_experts - self.num_shared)
            self.a = rng.normal(size=self.num_experts)
            self.b = rng.normal(size=self.num_experts)
            self.sigma = np.ones(self.num_experts)

    def compute_gating_probs(self, X):
        """
        Compute the gating probabilities for each expert using top-K gating.

        Parameters
        ----------
        X : np.ndarray, shape (n,)

        Returns
        -------
        gating_probs : np.ndarray, shape (n, num_experts)
            Gating probabilities for each expert.
        """
        n = len(X)
        # Vector of shape (n, num_experts) with logits for each data point and expert
        logits = np.outer(X, self.beta1) + self.beta0  # shape (n, num_experts)

        # Initialize gating probabilities with zeros
        gating_probs = np.zeros_like(logits)

        # Find top-K indices for each row in logits
        # argpartition is O(k), better than a full sort O(k log k)
        topk_indices = np.argpartition(logits, -self.K, axis=1)[:, -self.K:]  # shape (n, K)

        # Create a mask for the top-K indices
        mask = np.zeros_like(logits, dtype=bool)
        rows = np.arange(n)[:, np.newaxis]
        mask[rows, topk_indices] = True

        # Apply mask to get only the top-K logits, others will be -inf
        masked_logits = np.where(mask, logits, -np.inf)

        if self.router_type == "softmax":
            # Compute softmax over the masked logits
            # Shift for numerical stability
            shifted_logits = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            gating_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        elif self.router_type == "sigmoid":
            sigmoid_logits = 1 / (1 + np.exp(-masked_logits))
            gating_probs = sigmoid_logits / np.sum(sigmoid_logits, axis=1, keepdims=True)
        else:
            raise ValueError("Invalid router type. Choose 'softmax' or 'sigmoid'.")

        if self.num_shared > 0: # add shared experts with weights = 1/num_shared
            gating_probs = np.concatenate([gating_probs, np.ones_like(gating_probs[:, :self.num_shared]) / self.num_shared], axis=1)

        return gating_probs

    def compute_expert_likelihoods(self, X, Y):
        """
        Compute the likelihood of each data point under each expert (vectorized).

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)

        Returns
        -------
        likelihoods : np.ndarray, shape (n, num_experts)
            Likelihood of each data point under each expert.
        """
        n = len(X)
        num_experts = self.num_experts

        # Reshape inputs for broadcasting
        X_reshaped = X.reshape(n, 1)  # shape (n, 1)
        Y_reshaped = Y.reshape(n, 1)  # shape (n, 1)            
        a_reshaped = self.a.reshape(1, num_experts)  # shape (1, num_experts)
        b_reshaped = self.b.reshape(1, num_experts)  # shape (1, num_experts)
        sigma_reshaped = self.sigma.reshape(1, num_experts)  # shape (1, num_experts)

        # Calculate means
        means = X_reshaped * a_reshaped + b_reshaped  # shape (n, num_experts)

        # Compute the normal PDF manually for efficiency
        # norm.pdf(x, loc=mu, scale=sigma) = 1/(sqrt(2π)*σ)*exp(-0.5*((x - mu)/σ)^2)

        # Compute (Y - mean) / sigma for each data point and expert
        diff = (Y_reshaped - means) / sigma_reshaped  # shape (n, num_experts)

        # denominator = sqrt(2π)*sigma
        denom = np.sqrt(2 * np.pi) * sigma_reshaped  # shape (1, num_experts)

        # pdf values
        likelihoods = (1.0 / denom) * np.exp(-0.5 * diff**2)  # shape (n, num_experts)

        return likelihoods

    def e_step(self, X, Y):
        """
        Perform the E-step of the EM algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)

        Returns
        -------
        responsibilities : np.ndarray, shape (n, num_experts)
            Responsibility of each expert for each data point.
        gating_probs : np.ndarray, shape (n, num_experts)
            Gating probabilities (reused in M-step).
        """
        # Compute gating probabilities
        gating_probs = self.compute_gating_probs(X)

        # Compute expert likelihoods
        likelihoods = self.compute_expert_likelihoods(X, Y)

        # Compute unnormalized responsibilities
        responsibilities = gating_probs * likelihoods

        # Normalize responsibilities across experts
        row_sums = np.sum(responsibilities, axis=1, keepdims=True) + 1e-10
        responsibilities /= row_sums

        return responsibilities, gating_probs

    def m_step(self, X, Y, responsibilities, gating_probs, learning_rate=0.01):
        """
        Perform the M-step of the EM algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)
        responsibilities : np.ndarray, shape (n, num_experts)
            Responsibility of each expert for each data point.
        gating_probs : np.ndarray, shape (n, num_experts)
            Gating probabilities (from the E-step, reused here).
        learning_rate : float
            Learning rate for gradient ascent when updating gating parameters.
        """
        # Update expert parameters - vectorized implementation
        n = len(X)
        num_experts = self.num_experts

        # Reshape inputs for broadcasting
        X_reshaped = X.reshape(n, 1)  # shape (n, 1)
        Y_reshaped = Y.reshape(n, 1)  # shape (n, 1)

        # Sum responsibilities for each expert
        r_sum = np.sum(responsibilities, axis=0)  # shape (num_experts,)
        mask = r_sum > 1e-10

        if np.any(mask):
            # Weighted sums
            X_weighted_sum = np.sum(responsibilities * X_reshaped, axis=0)  # shape (num_experts,)
            Y_weighted_sum = np.sum(responsibilities * Y_reshaped, axis=0)  # shape (num_experts,)

            # Weighted means
            X_mean = np.zeros(num_experts)
            Y_mean = np.zeros(num_experts)
            
            X_mean[mask] = X_weighted_sum[mask] / r_sum[mask]
            Y_mean[mask] = Y_weighted_sum[mask] / r_sum[mask]

            # Reshape means for broadcasting
            X_mean_reshaped = X_mean.reshape(1, num_experts)  # shape (1, num_experts)
            Y_mean_reshaped = Y_mean.reshape(1, num_experts)  # shape (1, num_experts)

            # Compute centered values
            X_centered = X_reshaped - X_mean_reshaped  # shape (n, num_experts)
            Y_centered = Y_reshaped - Y_mean_reshaped  # shape (n, num_experts)

            # Weighted covariance
            cov = np.sum(responsibilities * X_centered * Y_centered, axis=0) / np.maximum(r_sum, 1e-10)  # shape (num_experts,)

            # Weighted variance
            var_X = np.sum(responsibilities * X_centered**2, axis=0) / np.maximum(r_sum, 1e-10)  # shape (num_experts,)

            # Update a and b for routed experts
            var_mask = var_X > 1e-10
            self.a[var_mask] = cov[var_mask] / var_X[var_mask]
            self.b[var_mask] = Y_mean[var_mask] - self.a[var_mask] * X_mean[var_mask]

            a_reshaped = self.a.reshape(1, num_experts)  # shape (1, num_experts)
            b_reshaped = self.b.reshape(1, num_experts)  # shape (1, num_experts)
            
            # Calculate residuals after applying regular experts
            means = X_reshaped * a_reshaped + b_reshaped  # shape (n, num_experts)
            
            # Compute residuals for sigma update
            residuals = Y_reshaped - means  # shape (n, num_experts)

            # Compute sigma for routed experts

            self.sigma = np.sqrt(
                np.sum(responsibilities[:, var_mask] * residuals[:, var_mask]**2, axis=0) /
                np.maximum(r_sum[var_mask], 1e-10)
            )

        # breakpoint()
        # Update gating parameters with gradient ascent
        differences = (responsibilities - gating_probs)[:, :self.num_experts - self.num_shared]  # shape (n, num_routed_experts)

        # Compute gradients
        grad_beta0 = np.sum(differences, axis=0)  # shape (num_experts,)
        grad_beta1 = np.sum(differences * X[:, None], axis=0)  # shape (num_experts,)

        # Apply gradients
        self.beta0 += learning_rate * grad_beta0
        self.beta1 += learning_rate * grad_beta1
