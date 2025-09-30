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
            'sigma': []
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

        # Compute expert likelihoods
        likelihoods = self.model.compute_expert_likelihoods(X, Y)

        # Compute log-likelihood
        weighted_likelihoods = gating_probs * likelihoods
        point_log_likelihoods = np.log(np.sum(weighted_likelihoods, axis=1) + 1e-10)
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
            Tuple of (beta1_true, beta0_true, a_true, b_true, sigma_true)

        Returns
        -------
        self
        """
        # Initialize model parameters
        self.model.initialize_parameters(X, Y, true_params)

        # Initialize log-likelihood
        prev_log_likelihood = -np.inf

        # EM algorithm
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            # E-step
            responsibilities = self.model.e_step(X, Y)

            # M-step
            self.model.m_step(X, Y, responsibilities, learning_rate=self.learning_rate)

            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(X, Y)

            # Store history
            self.history['log_likelihood'].append(log_likelihood)
            self.history['beta0'].append(self.model.beta0.copy())
            self.history['beta1'].append(self.model.beta1.copy())
            self.history['a'].append(self.model.a.copy())
            self.history['b'].append(self.model.b.copy())
            self.history['sigma'].append(self.model.sigma.copy())

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                break

            prev_log_likelihood = log_likelihood

        return self

    def plot_convergence(self, filename='convergence.pdf'):
        """
        Plot the convergence of the log-likelihood.

        Parameters
        ----------
        filename : str
            Name of the file to save the plot to.
        """
        # Ensure the export directory exists
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
        # Ensure the export directory exists
        export_dir = 'export'
        os.makedirs(export_dir, exist_ok=True)

        param_names = ['beta0', 'beta1', 'a', 'b', 'sigma']

        for param_name in param_names:
            plt.figure(figsize=(10, 6))
            param_history = np.array(self.history[param_name])

            for j in range(self.model.k):
                plt.plot(param_history[:, j], label=f'Expert {j+1}')

            plt.xlabel('Iteration')
            plt.ylabel(param_name)
            plt.title(f'Convergence of {param_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(export_dir, f'{filename_prefix}_{param_name}.pdf'))
            plt.close()
