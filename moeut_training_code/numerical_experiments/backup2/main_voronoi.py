import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from IPython import embed
from voronoi_experiment import run_voronoi_style_error_experiment


def main():
    """
    Main entry point for the Voronoi-style error experiment.
    """
    # Record start time
    start_time = time.time()

    # Define ground truth parameters
    ground_truth_params = {
        "beta0_true": np.array([-8.0, 0.0]),
        "beta1_true": np.array([25.0, 0.0]),
        "a_true": np.array([-20.0, 20.0]),
        "b_true": np.array([15.0, -5.0]),
        "sigma_true": np.array([0.3, 0.4]),
        "a_shared_true": np.array([2.0]),
        "b_shared_true": np.array([-3.0]),
        "sigma_shared_true": np.array([0.5]),
    }

    # Define sample sizes
    # sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 80000, 100000]
    sample_sizes = list(np.concatenate([np.linspace(100, 1000, 10, dtype=int), np.linspace(1500, 10000, 30, dtype=int), np.linspace(15000, 100000, 60, dtype=int)])) #  for the experiment
    
    # sample_sizes = np.linspace(100, 1000, 10, dtype=int)  # Sample sizes 

    # model parameters
    num_experts_true = len(ground_truth_params["beta0_true"])  # True number of experts
    num_experts_fit = 2                 # Number of experts to fit
    K_true = 1                          # Number of experts to keep in the true gating function
    K_fit = 1                           # Number of experts to keep in the fitted gating function
    num_shared = 0                      # Number of shared experts
    router_type = "sigmoid"             # Router type for the gating function
    # training parameters
    n_trials = 40                       # Number of trials for each sample size
    max_iter = 2000                     # Maximum number of iterations for the EM algorithm
    tol = 1e-6                          # Tolerance for convergence
    lr = 5e-3                           # Learning rate for gradient ascent
    min_n_processor = 16                # Minimum number of processors to use
    
    if num_shared == 0:
        ground_truth_params["a_shared_true"] = np.array([0.0])
        ground_truth_params["b_shared_true"] = np.array([0.0])
        ground_truth_params["sigma_shared_true"] = np.array([0.0])

    # Run the experiment
    print("Running Voronoi-style error experiment...")
    sample_sizes, avg_errors, std_errors = run_voronoi_style_error_experiment(
        ground_truth_params,
        sample_sizes, num_experts_true, num_experts_fit, K_true, K_fit, router_type, num_shared,
        n_trials=n_trials, max_iter=max_iter, tol=tol, lr=lr, min_n_processor=min_n_processor,
        filename='voronoi_style_error.pdf'
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    print("Experiment completed successfully!")
    print(f"Results saved to export/voronoi_style_error.pdf")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    # Print the results
    print("\nResults:")
    print("Sample Size | Average Error | Standard Error")
    print("-" * 45)
    for n, avg, std in zip(sample_sizes, avg_errors, std_errors):
        print(f"{n:11d} | {avg:13.6f} | {std:14.6f}")
        
    # save the results to a file
    np.save(f"export/results.npy", (sample_sizes, avg_errors, std_errors))

if __name__ == "__main__":
    main()
