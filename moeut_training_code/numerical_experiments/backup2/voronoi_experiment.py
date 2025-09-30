import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from IPython import embed

from data_generator import generate_data, visualize_data
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
    num_experts_fit = len(beta0_hat)
    num_experts_true = len(beta0_true)

    # Vectorized implementation
    # For each fitted component i, compute distance to all true components j
    # Shape: (num_experts_fit, num_experts_true)
    beta1_dists = np.abs(beta0_hat[:, np.newaxis] - beta0_true[np.newaxis, :])
    a_dists = np.abs(a_hat[:, np.newaxis] - a_true[np.newaxis, :])
    b_dists = np.abs(b_hat[:, np.newaxis] - b_true[np.newaxis, :])
    sigma_dists = np.abs(sigma_hat[:, np.newaxis] - sigma_true[np.newaxis, :])

    # Total distance for each (i,j) pair
    total_dists = beta1_dists + a_dists + b_dists + sigma_dists

    # Find minimum distance for each fitted component i
    min_dists = np.min(total_dists, axis=1)

    # Weight by exp(beta0_hat[i])
    weights = np.exp(beta0_hat)

    # Compute weighted sum
    total_err = np.sum(weights * min_dists)

    return total_err




def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                           beta0_true, beta1_true, a_true, b_true, sigma_true):
    # ‑‑ parameter‑distance matrix (scalar parameters shown; extend to vectors if needed)
    d_beta1 = np.abs(beta1_hat[:, np.newaxis] - beta1_true[np.newaxis, :])
    d_a     = np.abs(a_hat[:, np.newaxis]     - a_true[np.newaxis, :])
    d_b     = np.abs(b_hat[:, np.newaxis]     - b_true[np.newaxis, :])
    d_sigma = np.abs(sigma_hat[:, np.newaxis] - sigma_true[np.newaxis, :])
    d_total = d_beta1 + d_a + d_b + d_sigma
    min_dist = np.min(d_total, axis=1)
    
    # parameter mismatch term
    param_err = np.sum(np.exp(beta0_hat) * min_dist)

    # compute mass mismatch term
    mask = (np.argmin(d_total, axis=1) == 1)
    exp_beta0_assigned = np.zeros_like(beta0_hat)
    exp_beta0_assigned[mask] = np.exp(beta0_hat[mask])
    
    exp_beta0_true_assigned = np.zeros_like(beta0_true)
    exp_beta0_true_assigned[mask] = np.exp(beta0_true[mask])
    
    # compute mass mismatch term based on assign position
    mass_err = np.sum(np.abs(exp_beta0_assigned - exp_beta0_true_assigned))

    return param_err + mass_err



def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                           beta0_true, beta1_true, a_true, b_true, sigma_true):
    # ‑‑ parameter‑distance matrix (scalar parameters shown; extend to vectors if needed)
    d_beta1 = np.abs(beta1_hat[:, np.newaxis] - beta1_true[np.newaxis, :])
    d_a     = np.abs(a_hat[:, np.newaxis]     - a_true[np.newaxis, :])
    d_b     = np.abs(b_hat[:, np.newaxis]     - b_true[np.newaxis, :])
    d_sigma = np.abs(sigma_hat[:, np.newaxis] - sigma_true[np.newaxis, :])
    d_total = d_beta1 + d_a + d_b + d_sigma
    min_dist = np.min(d_total, axis=1)
    
    # parameter mismatch term
    param_err = np.sum(np.exp(beta0_hat) * min_dist)

    # compute mass mismatch term
    mask = (np.argmin(d_total, axis=1) == 1)
    exp_beta0_assigned = np.zeros_like(beta0_hat)
    exp_beta0_assigned[mask] = np.exp(beta0_hat[mask])
    
    exp_beta0_true_assigned = np.zeros_like(beta0_true)
    exp_beta0_true_assigned[mask] = np.exp(beta0_true[mask])
    
    # compute mass mismatch term based on assign position
    mass_err = np.sum(np.abs(exp_beta0_assigned - exp_beta0_true_assigned))

    return param_err + mass_err


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
            plt.errorbar(sample_sizes, errors, yerr=std_errors, fmt='-o', color='blue', elinewidth=1, label='D₁(Ĝₙ, G*)')
        else:
            plt.plot(sample_sizes, errors, '-o', color='blue', label='D₁(Ĝₙ, G*)')
        
        # Use log scales for both axes to match the example
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log(sample size)')
        plt.ylabel('log(loss)')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.title(title)
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
        # plt.yscale('log')  # User prefers normal scale for Voronoi loss
        plt.xlabel('Sample Size (log scale)')
        plt.ylabel('Voronoi Loss')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(export_dir, filename))
        plt.close()


def run_single_trial(args):
    """
    Run a single trial for a given sample size.

    Parameters
    ----------
    args : tuple
        Tuple containing (n, beta0_true, beta1_true, a_true, b_true, sigma_true, num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared, max_iter, tol, lr)

    Returns
    -------
    err : float
        Voronoi-style error for this trial.
    """
    (n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, \
        num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared, max_iter, tol, lr) = args

    try:
        # Generate data - ensure n is an integer
        n = int(n)
        X, Y = generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, K_true, router_type)

        # Create and fit model
        model = MoEModel(num_experts=num_experts_fit, K=K_fit, router_type=router_type, num_shared=num_shared)
        trainer = MoETrainer(model, max_iter=max_iter, tol=tol, learning_rate=lr, verbose=False)

        # Pass true parameters for initialization
        true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true)
        trainer.fit(X, Y, true_params=true_params)

        if num_shared > 0:
            # Compute Voronoi-style error
            err = voronoi_style_error(
                model.beta0, model.beta1, model.a, model.b, model.sigma,
                beta0_true, beta1_true, np.concatenate([a_true, a_shared_true]), np.concatenate([b_true, b_shared_true]), np.concatenate([sigma_true, sigma_shared_true]), num_shared
            )
        else:
            # Compute Voronoi-style error
            err = voronoi_style_error(
                model.beta0, model.beta1, model.a, model.b, model.sigma,
                beta0_true, beta1_true, a_true, b_true, sigma_true
            )
        
        return err
    except Exception as e:
        print(f"Error in trial with n={n}: {str(e)}")
        # Return a default value in case of error
        return float('nan')


def run_voronoi_style_error_experiment(ground_truth_params,
                                       sample_sizes, num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared,
                                       n_trials=10, max_iter=2000, tol=1e-6, lr=5e-3, min_n_processor=8,
                                       filename='voronoi_style_error.pdf'):
    """
    Run an experiment to compute the Voronoi-style error for different sample sizes.

    Parameters
    ----------
    ground_truth_params : dict
        Dictionary containing the true parameters of the model.
    sample_sizes : list or np.ndarray
        List of sample sizes to run the experiment for.
    num_experts_true : int
        True number of experts.
    num_experts_fit : int
        Number of experts to fit.
    K_true : int
        Number of experts to keep in the true gating function.
    K_fit : int
        Number of experts to keep in the fitted gating function.
    router_type : str
        Router type for the gating function.
    num_shared : int
        Number of shared experts.
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
    # Set up multiprocessing
    n_processes = min(cpu_count(), min_n_processor)  # Use at most min_n_processor cores
    print(f"Using {n_processes} processes for parallel execution")
    
    # visualize_data with highest sample size
    max_sample_size = max(sample_sizes)
    X, Y = generate_data(max_sample_size, ground_truth_params["beta0_true"], ground_truth_params["beta1_true"], 
                         ground_truth_params["a_true"], ground_truth_params["b_true"], ground_truth_params["sigma_true"], 
                         ground_truth_params["a_shared_true"], ground_truth_params["b_shared_true"], ground_truth_params["sigma_shared_true"],
                         K_true, router_type)
    visualize_data(X, Y, filename='voronoi_x_y_relation.pdf')

    avg_errors = []
    std_errors = []

    for n in tqdm(sample_sizes, desc="Running experiment for different sample sizes"):
        # Prepare arguments for each trial
        args_list = [(int(n), ground_truth_params["beta0_true"], ground_truth_params["beta1_true"], ground_truth_params["a_true"], 
                      ground_truth_params["b_true"], ground_truth_params["sigma_true"], ground_truth_params["a_shared_true"], ground_truth_params["b_shared_true"], 
                      ground_truth_params["sigma_shared_true"], num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared, max_iter, tol, lr)
                    for _ in range(n_trials)]

        # Run trials in parallel
        with Pool(processes=n_processes) as pool:
            errors = list(tqdm(
                pool.imap(run_single_trial, args_list),
                total=n_trials,
                desc=f"Running trials for n={n}",
                leave=False
            ))
        
        # run without multiprocessing
        # errors = [run_single_trial(args) for args in args_list]

        # Filter out any NaN values from failed trials
        valid_errors = [e for e in errors if not np.isnan(e)]

        if len(valid_errors) > 0:
            avg_errors.append(np.mean(valid_errors))
            std_errors.append(np.std(valid_errors) if len(valid_errors) > 1 else 0)
        else:
            print(f"Warning: All trials failed for n={n}. Using placeholder values.")
            avg_errors.append(0.0)
            std_errors.append(0.0)

    # Plot the results
    plot_voronoi_style_error(sample_sizes, avg_errors, std_errors, filename=filename)

    return sample_sizes, avg_errors, std_errors