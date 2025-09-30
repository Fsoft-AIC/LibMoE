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
    num_shared = 0                      # Number of shared experts
    router_type = "softmax"             # Router type for the gating function
    max_iter: int = 1000             # Maximum number of iterations for the EM algorithm
    seed: int = 42                  # Random seed
    verbose: bool = False           # Enable verbose logging
    tol = 1e-6                      # Tolerance for convergence
    lr = 5e-3                       # Learning rate for gradient ascent
    
    # ground truth parameters
    beta0_true = np.array([-8.0, 0.0])
    beta1_true = np.array([25.0, 0.0])
    a_true     = np.array([-20.0, 20.0])
    b_true     = np.array([15.0, -5.0])
    sigma_true = np.array([0.3, 0.4])
    a_shared_true = np.array([-3.0])
    b_shared_true = np.array([2.0])
    sigma_shared_true = np.array([0.5])
    
    if num_shared == 0:
        a_shared_true = np.array([0.0])
        b_shared_true = np.array([0.0])
        sigma_shared_true = np.array([0.0])

    # sample size to run 
    sample_sizes = 200


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

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
