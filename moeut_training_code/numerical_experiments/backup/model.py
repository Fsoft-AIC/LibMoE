import numpy as np
from scipy.stats import norm
from numpy.random import default_rng
import matplotlib.pyplot as plt
import os


class MoEModel:
    def __init__(self, k, K):
        """
        Initialize a Mixture of Experts model.

        Parameters
        ----------
        k : int
            Number of experts.
        K : int
            Number of experts to keep in the gating function.
        """
        self.k = k
        self.K = K

        # Initialize parameters
        self.beta0 = None
        self.beta1 = None
        self.a = None
        self.b = None
        self.sigma = None

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
        n_components = self.k
        n_features = 1

        (beta1_true, beta0_true, a_true, b_true, sigma_true) = true_params

        beta1_init = np.zeros((n_components, n_features))
        beta0_init = np.zeros(n_components)
        a_init = np.zeros((n_components, n_features))
        b_init = np.zeros(n_components)
        sigma_init = np.zeros(n_components)

        for i in range(n_components):
            i_true = np.random.randint(0, beta0_true.size) if i > beta0_true.size - 1 else i

            # add a larger noise
            noise_scale = 0.005 * n_samples**(-0.083)
            sigma_noise_scale = 0.005 * n_samples**(-0.25)
            beta1_init[i] = beta1_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            beta0_init[i] = beta0_true[i_true] + np.random.normal(0, noise_scale)
            a_init[i] = a_true[i_true] + np.random.normal(0, noise_scale, size=(n_features))
            b_init[i] = b_true[i_true] + np.random.normal(0, noise_scale)
            sigma_init[i] = sigma_true[i_true] + np.abs(np.random.normal(0, sigma_noise_scale))

        return (beta1_init.flatten(), beta0_init, a_init.flatten(), b_init, sigma_init)

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

            # Initialize gating parameters
            self.beta0 = np.random.randn(self.k)
            self.beta1 = np.random.randn(self.k)

            # Initialize expert parameters
            self.a = np.random.randn(self.k)
            self.b = np.random.randn(self.k)
            self.sigma = np.ones(self.k)



    def compute_gating_probs(self, X):
        """
        Compute the gating probabilities for each expert.

        Parameters
        ----------
        X : np.ndarray, shape (n,)

        Returns
        -------
        gating_probs : np.ndarray, shape (n, k)
            Gating probabilities for each expert.
        """
        n = len(X)
        gating_probs = np.zeros((n, self.k))

        for i in range(n):
            # Compute logits
            logits = self.beta1 * X[i] + self.beta0

            # Identify top-K indices
            topk_idx = np.argsort(logits)[-self.K:]

            # Initialize gating probabilities to 0
            gating_probs[i, :] = 0

            # Compute softmax over top-K experts
            topk_logits = logits[topk_idx]
            w = np.exp(topk_logits - np.max(topk_logits))
            w /= np.sum(w)

            # Set gating probabilities for top-K experts
            gating_probs[i, topk_idx] = w

        return gating_probs

    def compute_expert_likelihoods(self, X, Y):
        """
        Compute the likelihood of each data point under each expert.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)

        Returns
        -------
        likelihoods : np.ndarray, shape (n, k)
            Likelihood of each data point under each expert.
        """
        n = len(X)
        likelihoods = np.zeros((n, self.k))

        for j in range(self.k):
            # Compute mean and standard deviation for expert j
            mean = self.a[j] * X + self.b[j]
            stdev = self.sigma[j]

            # Compute likelihood
            likelihoods[:, j] = norm.pdf(Y, mean, stdev)

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
        responsibilities : np.ndarray, shape (n, k)
            Responsibility of each expert for each data point.
        """
        # Compute gating probabilities
        gating_probs = self.compute_gating_probs(X)

        # Compute expert likelihoods
        likelihoods = self.compute_expert_likelihoods(X, Y)

        # Compute responsibilities
        responsibilities = gating_probs * likelihoods

        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / (row_sums + 1e-10)

        return responsibilities

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

        # Update expert parameters
        for j in range(self.k):
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
        for j in range(self.k):
            grad_beta0 = np.sum(responsibilities[:, j] - self.compute_gating_probs(X)[:, j])
            grad_beta1 = np.sum((responsibilities[:, j] - self.compute_gating_probs(X)[:, j]) * X)

            self.beta0[j] += learning_rate * grad_beta0
            self.beta1[j] += learning_rate * grad_beta1
