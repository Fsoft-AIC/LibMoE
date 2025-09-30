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


def plot_voronoi_style_error(sample_sizes, errors, std_errors=None, filename='voronoi_style_error.pdf', title='Voronoi-Style Error vs. Sample Size'):

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
    (n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true, \
        num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared, max_iter, tol, lr, type_voronoi) = args

    try:
        # Generate data - ensure n is an integer
        n = int(n)
        X, Y = generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true, K_true, router_type)

        # Create and fit model
        model = MoEModel(num_experts=num_experts_fit, K=K_fit, router_type=router_type, num_shared=num_shared)
        trainer = MoETrainer(model, max_iter=max_iter, tol=tol, learning_rate=lr, verbose=False)

        # Pass true parameters for initialization
        true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true)
        trainer.fit(X, Y, true_params=true_params)

        if type_voronoi == "d2":
            from voronoi_error.d2_voronoi_err import voronoi_D2_linear
            err = voronoi_D2_linear(
                omega_hat=model.omega_shared,
                kappa1_hat=model.a_shared,
                kappa0_hat=model.b_shared,
                tau_hat=model.sigma_shared,
                omega_true=omega_shared_true,
                kappa1_true=a_shared_true,
                kappa0_true=b_shared_true,
                tau_true=sigma_shared_true,
                
                beta0_hat=model.beta0,
                beta1_hat=model.beta1,
                eta1_hat=model.a,
                eta0_hat=model.b,
                nu_hat=model.sigma,
                beta0_true=beta0_true,
                beta1_true=beta1_true,
                eta1_true=a_true,
                eta0_true=b_true,
                nu_true=sigma_true,
            )
        else:
            # raise not implemented error
            raise NotImplementedError("Only d2 voronoi error is implemented")
        
        return err
    except Exception as e:
        print(f"Error in trial with n={n}: {str(e)}")
        # Return a default value in case of error
        return float('nan')


def run_voronoi_style_error_experiment(ground_truth_params,
                                       sample_sizes, num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared,
                                       n_trials=10, max_iter=2000, tol=1e-6, lr=5e-3, min_n_processor=8,
                                       filename='voronoi_style_error.pdf', type_voronoi="d2"):

    # Set up multiprocessing
    n_processes = min(cpu_count(), min_n_processor)  # Use at most min_n_processor cores
    print(f"Using {n_processes} processes for parallel execution")
    
    # visualize_data with highest sample size
    max_sample_size = 500
    X, Y = generate_data(max_sample_size, ground_truth_params["beta0_true"], ground_truth_params["beta1_true"], 
                         ground_truth_params["a_true"], ground_truth_params["b_true"], ground_truth_params["sigma_true"], 
                         ground_truth_params["a_shared_true"], ground_truth_params["b_shared_true"], ground_truth_params["sigma_shared_true"], 
                         ground_truth_params["omega_shared_true"], K_true, router_type)
    visualize_data(X, Y, filename='voronoi_x_y_relation.pdf')

    avg_errors = []
    std_errors = []

    for n in tqdm(sample_sizes, desc="Running experiment for different sample sizes"):
        # Prepare arguments for each trial
        args_list = [(int(n), ground_truth_params["beta0_true"], ground_truth_params["beta1_true"], ground_truth_params["a_true"], ground_truth_params["b_true"], 
                      ground_truth_params["sigma_true"], ground_truth_params["a_shared_true"], ground_truth_params["b_shared_true"], ground_truth_params["sigma_shared_true"], 
                      ground_truth_params["omega_shared_true"], num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared, max_iter, tol, lr, type_voronoi)
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