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
            Number of routed experts.
        K : int
            Number of experts to keep in the gating function.
        num_shared : int
            Number of shared experts
        router_type : str
            Type of router to use, either 'softmax' or 'sigmoid'.
        """
        self.num_experts = num_experts  # number of routed experts
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
        self.a_shared = None
        self.b_shared = None
        self.sigma_shared = None
        self.omega_shared = None

    def init_params(self, true_params, n_samples):
        """
        Starting values for EM algorithm based on true parameters.

        Parameters
        ----------
        true_params : tuple
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true)
        n_samples : int
            Number of samples in the dataset

        Returns
        -------
        tuple
            Tuple of (beta1_init, beta0_init, a_init, b_init, sigma_init, a_shared_init, b_shared_init, sigma_shared_init, omega_shared_init)
        """
        (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true) = true_params
        
        # add a smaller noise
        noise_scale = 0.000005 * n_samples**(-0.083)
        sigma_noise_scale = 0.000005 * n_samples**(-0.25)

        n_components = self.num_experts
        n_features = 1
        # for routed experts
        beta1_init = np.zeros((n_components, n_features))
        beta0_init = np.zeros(n_components)
        a_init = np.zeros((n_components, n_features))
        b_init = np.zeros(n_components)
        sigma_init = np.zeros(n_components)

        for i in range(n_components):
            i_true = np.random.randint(0, beta0_true.size) if i > beta0_true.size - 1 else i

            beta1_init[i] = beta1_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            beta0_init[i] = beta0_true[i_true] + np.random.normal(0, noise_scale)
            a_init[i] = a_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            b_init[i] = b_true[i_true] + np.random.normal(0, noise_scale)
            sigma_init[i] = sigma_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))

        # for shared experts
        n_shared_component = self.num_shared
        n_shared_features = 1

        a_shared_init = np.zeros((n_shared_component, n_shared_features))
        b_shared_init = np.zeros(n_shared_component)
        sigma_shared_init = np.zeros(n_shared_component)
        omega_shared_init = np.zeros(n_shared_component)

        for i in range(n_shared_component):
            i_true = np.random.randint(0, a_shared_true.size) if i > a_shared_true.size - 1 else i
            a_shared_init[i] = a_shared_true[i_true] + np.random.normal(0, noise_scale, size=(n_shared_features))
            b_shared_init[i] = b_shared_true[i_true] + np.random.normal(0, noise_scale)
            sigma_shared_init[i] = sigma_shared_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))
            omega_shared_init[i] = omega_shared_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))

        return beta1_init.flatten(), beta0_init, a_init.flatten(), b_init, sigma_init, a_shared_init.flatten(), b_shared_init, sigma_shared_init, omega_shared_init


    def initialize_parameters(self, X, Y, true_params=None):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)
        true_params : tuple, optional
            If provided, initialize parameters based on true parameters.
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true)
        """
        n = len(X)

        if true_params is not None:
            # Initialize parameters based on true parameters
            beta1_init, beta0_init, a_init, b_init, sigma_init, a_shared_init, b_shared_init, sigma_shared_init, omega_shared_init = self.init_params(true_params, n)
            self.beta0 = beta0_init
            self.beta1 = beta1_init
            self.a = a_init
            self.b = b_init
            self.sigma = sigma_init
            self.a_shared = a_shared_init
            self.b_shared = b_shared_init
            self.sigma_shared = sigma_shared_init
            self.omega_shared = omega_shared_init
        else:
            # Initialize with random values
            rng = default_rng()
            self.beta0 = rng.normal(size=self.num_experts - self.num_shared)
            self.beta1 = rng.normal(size=self.num_experts - self.num_shared)
            self.a = rng.normal(size=self.num_experts)
            self.b = rng.normal(size=self.num_experts)
            self.sigma = np.ones(self.num_experts)
            self.a_shared = rng.normal(size=self.num_shared)
            self.b_shared = rng.normal(size=self.num_shared)
            self.sigma_shared = np.ones(self.num_shared)
            self.omega_shared = np.ones(self.num_shared)

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
        
        # for routed experts
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
        
        # for shared experts
        a_shared_reshaped = self.a_shared.reshape(1, self.num_shared)  # shape (1, num_shared)
        b_shared_reshaped = self.b_shared.reshape(1, self.num_shared)  # shape (1, num_shared)
        
        # check if sigma_shared is > 1e10:
        # if np.any(self.sigma_shared > 1e10):
        sigma_shared_reshaped = self.sigma_shared.reshape(1, self.num_shared)  # shape (1, num_shared)
        means_shared = X_reshaped * a_shared_reshaped + b_shared_reshaped  # shape (n, num_shared)
        
        diff_shared = (Y_reshaped - means_shared) / sigma_shared_reshaped  # shape (n, num_shared)
        # denominator = sqrt(2π)*sigma
        denom_shared = np.sqrt(2 * np.pi) * sigma_shared_reshaped  # shape (1, num_shared)
        likelihoods_shared = (1.0 / denom_shared) * np.exp(-0.5 * diff_shared**2)  # shape (n, num_shared)
        # else:
        #     # set likelihoods to 0
        #     likelihoods_shared = np.zeros((n, self.num_shared))
        
        likelihoods = np.concatenate([likelihoods, likelihoods_shared], axis=1)

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
        # shared probs: size [X.shape[0], num_shared], values are self.omega_shared (with shape [num_shared])
        shared_probs = np.tile(self.omega_shared, (X.shape[0], 1))
        # add the weight of shared experts to gating_probs
        gating_probs = np.concatenate([gating_probs, shared_probs], axis=1)

        # Compute expert likelihoods
        likelihoods = self.compute_expert_likelihoods(X, Y)

        # Compute unnormalized responsibilities
        responsibilities = gating_probs * likelihoods

        # Normalize responsibilities across experts
        row_sums = np.sum(responsibilities, axis=1, keepdims=True) + 1e-10
        responsibilities /= row_sums

        return responsibilities, gating_probs



    def m_step(self, X, Y, responsibilities, learning_rate=0.01):
        """
        Perform the M-step of the EM algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)
        responsibilities : np.ndarray, shape (n, k)
            Responsibility of each expert for each data point.
        learning_rate : float
            Learning rate for gradient ascent when updating gating parameters.
        """
        n = len(X)
        num_total_experts = self.num_experts + self.num_shared

        # Update expert parameters in royted experts
        for j in range(self.num_experts):
            # Compute weighted sum of responsibilities
            r_sum = np.sum(responsibilities[:, j])

            if r_sum > 1e-10:
                # Update a and b using weighted linear regression
                X_weighted = X * responsibilities[:, j]
                Y_weighted = Y * responsibilities[:, j]

                # Compute weighted means
                X_mean = np.sum(X_weighted) / r_sum
                Y_mean = np.sum(Y_weighted) / r_sum

                # Compute weighted covariance
                cov = np.sum(responsibilities[:, j] * (X - X_mean) * (Y - Y_mean)) / r_sum
                var_X = np.sum(responsibilities[:, j] * (X - X_mean)**2) / r_sum

                if var_X > 1e-10:
                    self.a[j] = cov / var_X
                    self.b[j] = Y_mean - self.a[j] * X_mean

                    # Update sigma
                    residuals = Y - (self.a[j] * X + self.b[j])
                    self.sigma[j] = np.sqrt(np.sum(responsibilities[:, j] * residuals**2) / r_sum)

        # Update gating parameters using gradient ascent
        for j in range(self.num_experts):
            grad_beta0 = np.sum(responsibilities[:, j] - self.compute_gating_probs(X)[:, j])
            grad_beta1 = np.sum((responsibilities[:, j] - self.compute_gating_probs(X)[:, j]) * X)

            self.beta0[j] += learning_rate * grad_beta0
            self.beta1[j] += learning_rate * grad_beta1
            
        # update shared experts
        for j in range(self.num_experts, num_total_experts):
            # Compute weighted sum of responsibilities
            r_sum = np.sum(responsibilities[:, j])
            if r_sum > 1e-10:
                # Update a and b using weighted linear regression
                X_weighted = X * responsibilities[:, j]
                Y_weighted = Y * responsibilities[:, j]

                # Compute weighted means
                X_mean = np.sum(X_weighted) / r_sum
                Y_mean = np.sum(Y_weighted) / r_sum

                # Compute weighted covariance
                cov = np.sum(responsibilities[:, j] * (X - X_mean) * (Y - Y_mean)) / r_sum
                var_X = np.sum(responsibilities[:, j] * (X - X_mean)**2) / r_sum

                if var_X > 1e-10:
                    self.a_shared[j - self.num_experts] = cov / var_X
                    self.b_shared[j - self.num_experts] = Y_mean - self.a_shared[j - self.num_experts] * X_mean

                    # Update sigma
                    residuals = Y - (self.a_shared[j - self.num_experts] * X + self.b_shared[j - self.num_experts])
                    self.sigma_shared[j - self.num_experts] = np.sqrt(np.sum(responsibilities[:, j] * residuals**2) / r_sum)
                    
                    # update self.omega_shared (weights of shared experts)
                    self.omega_shared[j - self.num_experts] = r_sum / n
                    
                
                