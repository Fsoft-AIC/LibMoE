import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from data_generator import generate_data
from model import MoEModel
from trainer import MoETrainer


def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                       beta0_true, beta1_true, a_true, b_true, sigma_true):
    """
    A simplified 'Voronoi-style' metric. We do:
      1) For each fitted component i, find the ground-truth j that minimizes
         param-distance (like a Voronoi cell).
      2) Sum up (exp(beta0_hat[i])) * [|beta1_hat[i]-beta1_true[j]| + etc... ].
    This is not the paper's exact D1 or D2, but demonstrates the general shape
    of the metric used in Figure 3.
    """
    k_fit = len(beta0_hat)
    k_true = len(beta0_true)

    total_err = 0.0
    for i in range(k_fit):
        # pick j that is closest in parameter space
        best_j = 0
        best_dist = 1e15
        for j in range(k_true):
            d = (abs(beta1_hat[i] - beta1_true[j]) +
                 abs(a_hat[i] - a_true[j]) +
                 abs(b_hat[i] - b_true[j]) +
                 abs(sigma_hat[i] - sigma_true[j]))
            if d < best_dist:
                best_dist = d
                best_j = j
        # Weighted by exp(beta0_hat[i])
        w_i = np.exp(beta0_hat[i])
        total_err += w_i * best_dist
    return total_err


def plot_voronoi_style_error(sample_sizes, errors, std_errors=None, filename='voronoi_style_error.pdf', title='Voronoi-Style Error vs. Sample Size'):
    """
    Plot the Voronoi-style error as a function of sample size.

    Parameters
    ----------
    sample_sizes : list or np.ndarray
        List of sample sizes.
    errors : list or np.ndarray
        List of Voronoi-style errors corresponding to each sample size.
    std_errors : list or np.ndarray, optional
        Standard errors for each point, used for error bars.
    filename : str
        Name of the file to save the plot to.
    title : str
        Title of the plot.
    """
    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    # Convert to numpy arrays
    sample_sizes = np.array(sample_sizes)
    errors = np.array(errors)
    if std_errors is not None:
        std_errors = np.array(std_errors)

    # Fit a power law curve: y = a * x^b
    from scipy.optimize import curve_fit

    def power_law(x, a, b):
        return a * x**b

    try:
        # Fit the curve
        # breakpoint()
        params, _ = curve_fit(power_law, sample_sizes, errors)
        a, b = params

        # Generate points for the fitted curve
        x_fit = np.logspace(np.log10(min(sample_sizes)), np.log10(max(sample_sizes)), 100)
        y_fit = power_law(x_fit, a, b)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot the fitted curve
        plt.plot(x_fit, y_fit, '--', color='orange', label=f'{a:.2f}n^{b:.5f}')

        # Plot the data points with error bars if provided
        if std_errors is not None:
            plt.errorbar(sample_sizes, errors, yerr=std_errors, fmt='-o', color='blue', label='D₁(Ĝₙ, G*)')
        else:
            plt.plot(sample_sizes, errors, '-o', color='blue', label='D₁(Ĝₙ, G*)')

        # Use log scales for both axes to match the example
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log(sample size)')
        plt.ylabel('log(loss)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(export_dir, filename))
        plt.close()
    except Exception as e:
        # If curve fitting fails, create a simple plot without the fitted curve
        print(f"Warning: Could not fit curve. Error: {e}")
        plt.figure(figsize=(8, 6))

        if std_errors is not None:
            plt.errorbar(sample_sizes, errors, yerr=std_errors, fmt='-o', color='blue', label='D₁(Ĝₙ, G*)')
        else:
            plt.plot(sample_sizes, errors, '-o', color='blue', label='D₁(Ĝₙ, G*)')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log(sample size)')
        plt.ylabel('log(loss)')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(export_dir, filename))
        plt.close()


def run_voronoi_style_error_experiment(beta0_true, beta1_true, a_true, b_true, sigma_true,
                                      sample_sizes, k_true, k_fit, K_true, K_fit,
                                      n_trials=10, max_iter=2000, tol=1e-6, lr=5e-3, steps=10,
                                      filename='voronoi_style_error.pdf'):
    """
    Run an experiment to compute the Voronoi-style error for different sample sizes.

    Parameters
    ----------
    beta0_true, beta1_true, a_true, b_true, sigma_true : np.ndarray
        True parameters of the model.
    sample_sizes : list or np.ndarray
        List of sample sizes to run the experiment for.
    k_true : int
        True number of experts.
    k_fit : int
        Number of experts to fit.
    K_true : int
        Number of experts to keep in the true gating function.
    K_fit : int
        Number of experts to keep in the fitted gating function.
    n_trials : int
        Number of trials to run for each sample size.
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    tol : float
        Tolerance for convergence.
    lr : float
        Learning rate for gradient ascent.
    steps : int
        Number of gradient ascent steps.
    filename : str
        Name of the file to save the plot to.

    Returns
    -------
    sample_sizes : list
        List of sample sizes.
    avg_errors : list
        Average Voronoi-style error for each sample size.
    std_errors : list
        Standard deviation of Voronoi-style error for each sample size.
    """

    avg_errors = []
    std_errors = []

    for n in tqdm(sample_sizes, desc="Running experiment for different sample sizes"):
        errors = []

        for _ in tqdm(range(n_trials), desc=f"Running trials for n={n}", leave=False):
            # Generate data
            X, Y = generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K_true)

            # Create and fit model
            model = MoEModel(k=k_fit, K=K_fit)
            trainer = MoETrainer(model, max_iter=max_iter, tol=tol, learning_rate=lr, verbose=False)

            # Pass true parameters for initialization
            true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true)
            trainer.fit(X, Y, true_params=true_params)

            # Compute Voronoi-style error
            err = voronoi_style_error(
                model.beta0, model.beta1, model.a, model.b, model.sigma,
                beta0_true, beta1_true, a_true, b_true, sigma_true
            )
            errors.append(err)

        avg_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))

    # Plot the results
    plot_voronoi_style_error(sample_sizes, avg_errors, std_errors, filename=filename)

    return sample_sizes, avg_errors, std_errors