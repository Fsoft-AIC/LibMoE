import argparse
import logging
import numpy as np
from experiment import run_experiment
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    k_true: int = 2                 # True number of experts
    k_fit: int = 2                  # Number of experts to fit (for single mode)
    K: int = 1                      # Number of experts to keep in the gating function
    num_shared = 1                      # Number of shared experts
    router_type = "softmax"             # Router type for the gating function
    max_iter: int = 1000             # Maximum number of iterations for the EM algorithm
    seed: int = 42                  # Random seed
    verbose: bool = False           # Enable verbose logging
    tol = 1e-6                      # Tolerance for convergence
    lr = 5e-6                       # Learning rate for gradient ascent
    
    # ground truth parameters
    beta0_true        = np.array([ 0.0,  0.0])   # boundary at x=0
    beta1_true        = np.array([ 4.0, -4.0])   # expert 0 when x>0, expert 1 when x<0

    # Experts alone have slopes ±8
    a_true            = np.array([ 8.0, -8.0])
    b_true            = np.array([ 0.0,  0.0])
    sigma_true        = np.array([ 4.0,  3.0])    # moderate observation noise

    # Shared expert provides the “+5x +2” global trend
    a_shared_true     = np.array([ 2.0])         # slope +5
    b_shared_true     = np.array([ 5.0])         # intercept +2
    sigma_shared_true = np.array([ 5.0])         # a bit of shared‐expert noise
    omega_shared_true = np.array([ 1.0])

    
    if num_shared == 0:
        a_shared_true = np.array([0.0])
        b_shared_true = np.array([0.0])
        sigma_shared_true = np.array([0.0])
        omega_shared_true = np.array([0.0])

    # sample size to run 
    sample_sizes = 1000


def main():
    """
    Main entry point for the MoE experiments.
    """
    # Create config from command line arguments
    config = ExperimentConfig()

    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(f"Running experiment with {config.k_fit} experts...")
    results = run_experiment(config=config)
    
    # print result

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
