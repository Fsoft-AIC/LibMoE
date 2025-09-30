import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class MoETrainer:
    def __init__(self, model, max_iter=100, tol=1e-4, learning_rate=5e-3, verbose=True):
        """
        Initialize a trainer for the MoE model.

        Parameters
        ----------
        model : MoEModel
            The model to train.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        learning_rate : float
            Learning rate for gradient ascent in the M-step.
        verbose : bool
            Whether to print progress.
        """
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.history = {
            'log_likelihood': [],
            'beta0': [],
            'beta1': [],
            'a': [],
            'b': [],
            'sigma': [],
            'a_shared': [],
            'b_shared': [],
            'sigma_shared': [],
            'omega_shared': []
        }

    def compute_log_likelihood(self, X, Y):
        """
        Compute the log-likelihood of the data under the current model.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data.
        """
        # Compute gating probabilities
        gating_probs = self.model.compute_gating_probs(X)
        # shared probs: size [X.shape[0], num_shared], values are self.omega_shared (with shape [num_shared])
        shared_probs = np.tile(self.model.omega_shared, (X.shape[0], 1))
        gating_probs = np.concatenate([gating_probs, shared_probs], axis=1)

        # Compute expert likelihoods
        likelihoods = self.model.compute_expert_likelihoods(X, Y)

        # Compute log-likelihood using a more numerically stable approach
        # First multiply gating_probs and likelihoods
        weighted_likelihoods = gating_probs * likelihoods

        # Use logsumexp trick for numerical stability
        # log(sum(w_i * l_i)) = log(sum(exp(log(w_i * l_i)))) = logsumexp(log(w_i * l_i))
        # But we need to handle zeros in weighted_likelihoods
        with np.errstate(divide='ignore', invalid='ignore'):
            log_weighted = np.log(weighted_likelihoods)
            # Replace -inf (from log(0)) with a very negative number
            log_weighted[~np.isfinite(log_weighted)] = -1e10
            # Use the logsumexp trick
            max_log = np.max(log_weighted, axis=1, keepdims=True)
            exp_diff = np.exp(log_weighted - max_log)
            point_log_likelihoods = max_log.flatten() + np.log(np.sum(exp_diff, axis=1) + 1e-10)

        log_likelihood = np.sum(point_log_likelihoods)
        return log_likelihood

    def fit(self, X, Y, true_params=None):
        """
        Fit the model to the data using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n,)
        Y : np.ndarray, shape (n,)
        true_params : tuple, optional
            If provided, initialize parameters based on true parameters.
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true)

        Returns
        -------
        self
        """
        # Initialize model parameters
        self.model.initialize_parameters(X, Y, true_params)

        # Initialize log-likelihood
        prev_log_likelihood = -np.inf

        # Pre-allocate arrays for history to avoid repeated memory allocations
        self.history['log_likelihood'] = np.zeros(self.max_iter)
        self.history['beta0'] = np.zeros((self.max_iter, self.model.num_experts))
        self.history['beta1'] = np.zeros((self.max_iter, self.model.num_experts))
        self.history['a'] = np.zeros((self.max_iter, self.model.num_experts))
        self.history['b'] = np.zeros((self.max_iter, self.model.num_experts))
        self.history['sigma'] = np.zeros((self.max_iter, self.model.num_experts))
        self.history['a_shared'] = np.zeros((self.max_iter, self.model.num_shared))
        self.history['b_shared'] = np.zeros((self.max_iter, self.model.num_shared))
        self.history['sigma_shared'] = np.zeros((self.max_iter, self.model.num_shared))
        self.history['omega_shared'] = np.zeros((self.max_iter, self.model.num_shared))

        # EM algorithm
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            # E-step
            responsibilities, gating_probs = self.model.e_step(X, Y)

            # M-step
            self.model.m_step(X, Y, responsibilities, learning_rate=self.learning_rate)

            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(X, Y)
            # breakpoint()

            # Store history
            self.history['log_likelihood'][i] = log_likelihood
            self.history['beta0'][i] = self.model.beta0
            self.history['beta1'][i] = self.model.beta1
            self.history['a'][i] = self.model.a
            self.history['b'][i] = self.model.b
            self.history['sigma'][i] = self.model.sigma
            self.history['a_shared'][i] = self.model.a_shared
            self.history['b_shared'][i] = self.model.b_shared
            self.history['sigma_shared'][i] = self.model.sigma_shared
            self.history['omega_shared'][i] = self.model.omega_shared

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                # Trim history arrays to actual number of iterations
                for key in self.history:
                    self.history[key] = self.history[key][:i+1]
                break

            prev_log_likelihood = log_likelihood

        # If we didn't break early, trim history arrays to max_iter
        if i == self.max_iter - 1:
            for key in self.history:
                self.history[key] = self.history[key][:self.max_iter]

        return self

    def plot_convergence(self, filename='convergence.pdf'):
        """
        Plot the convergence of the log-likelihood.

        Parameters
        ----------
        filename : str
            Name of the file to save the plot to.
        """
        export_dir = 'export'
        os.makedirs(export_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['log_likelihood'])
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        plt.title('Convergence of Log-likelihood')
        plt.grid(True)
        plt.savefig(os.path.join(export_dir, filename))
        plt.close()

    def plot_parameter_convergence(self, filename_prefix='param_convergence'):
        """
        Plot the convergence of the parameters.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the filenames to save the plots to.
        """
        export_dir = 'export'
        os.makedirs(export_dir, exist_ok=True)

        # Plot regular expert parameters
        param_names = ['beta0', 'beta1', 'a', 'b', 'sigma', 'a_shared', 'b_shared', 'sigma_shared', 'omega_shared']
        for param_name in param_names:
            plt.figure(figsize=(10, 6))
            param_history = np.array(self.history[param_name])  # shape (iterations, num_experts)
            
            if "shared" not in param_name:
                for j in range(self.model.num_experts):
                    plt.plot(param_history[:, j], label=f'Expert {j+1}')
            else:
                for j in range(self.model.num_shared):
                    plt.plot(param_history[:, j], label=f'Shared Expert {j+1}')

            plt.xlabel('Iteration')
            plt.ylabel(param_name)
            plt.title(f'Convergence of {param_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(export_dir, f'{filename_prefix}_{param_name}.pdf'))
            plt.close()
