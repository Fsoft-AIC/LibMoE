import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from voronoi_experiment import run_voronoi_style_error_experiment


def main():
    """
    Main entry point for the Voronoi-style error experiment.
    """
    # Define ground truth parameters
    beta0_true = np.array([-8.0, 0.0])
    beta1_true = np.array([25.0, 0.0])
    a_true = np.array([-20.0, 20.0])
    b_true = np.array([15.0, -5.0])
    sigma_true = np.array([0.3, 0.4])
    
    # define the ground truth parameters with 3-dimensions
    # beta0_true = np.array([-8.0, 0.0, 1.0])
    # beta1_true = np.array([25.0, 0.0, 4.0])
    # a_true = np.array([-20.0, 20.0, 7.0])
    # b_true = np.array([15.0, -5.0, 10.0])
    # sigma_true = np.array([0.3, 0.4, 0.5])
    
    # Define sample sizes
    # sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    sample_sizes = list(np.linspace(100, 1000, 10, dtype=int))
    
    # Define experiment parameters
    k_true = len(beta0_true)  # True number of experts
    k_fit = 2                 # Number of experts to fit
    K_true = 1                # Number of experts to keep in the true gating function
    K_fit = 1                 # Number of experts to keep in the fitted gating function
    n_trials = 40             # Number of trials for each sample size
    max_iter = 2000           # Maximum number of iterations for the EM algorithm
    tol = 1e-4                # Tolerance for convergence
    lr = 5e-3                 # Learning rate for gradient ascent
    steps = 10                # Number of gradient ascent steps
    
    # Run the experiment
    print("Running Voronoi-style error experiment...")
    sample_sizes, avg_errors, std_errors = run_voronoi_style_error_experiment(
        beta0_true, beta1_true, a_true, b_true, sigma_true,
        sample_sizes, k_true, k_fit, K_true, K_fit,
        n_trials=n_trials, max_iter=max_iter, tol=tol, lr=lr, steps=steps,
        filename='voronoi_style_error.pdf'
    )
    
    print("Experiment completed successfully!")
    print(f"Results saved to gpt_code/export/voronoi_style_error.pdf")
    
    # Print the results
    print("\nResults:")
    print("Sample Size | Average Error | Standard Error")
    print("-" * 45)
    for n, avg, std in zip(sample_sizes, avg_errors, std_errors):
        print(f"{n:11d} | {avg:13.6f} | {std:14.6f}")

if __name__ == "__main__":
    main()
