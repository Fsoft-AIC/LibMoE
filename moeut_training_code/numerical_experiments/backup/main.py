import argparse
import logging
import numpy as np
from experiment import run_experiment, compare_models
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    mode: str = 'single'            # 'single' or 'compare'
    k_true: int = 2                 # True number of experts
    k_fit: int = 3                  # Number of experts to fit (for single mode)
    k_min: int = 2                  # Minimum number of experts to fit (for compare mode)
    k_max: int = 5                  # Maximum number of experts to fit (for compare mode)
    K: int = 2                      # Number of experts to keep in the gating function
    max_iter: int = 100             # Maximum number of iterations for the EM algorithm
    seed: int = 42                  # Random seed
    verbose: bool = False           # Enable verbose logging
    
    # ground truth parameters
    beta0_true = np.array([-8.0, 0.0])
    beta1_true = np.array([25.0, 0.0])
    a_true     = np.array([-20.0, 20.0])
    b_true     = np.array([15.0, -5.0])
    sigma_true = np.array([0.3, 0.4])
    
    # beta0_true = np.array([-8.0, 0.0, 1.0])
    # beta1_true = np.array([25.0, 0.0, 4.0])
    # a_true = np.array([-20.0, 20.0, 7.0])
    # b_true = np.array([15.0, -5.0, 10.0])
    # sigma_true = np.array([0.3, 0.4, 0.5])
    
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

    if config.mode == 'single':
        print(f"Running single experiment with {config.k_fit} experts...")
        results = run_experiment(config=config)
    else:
        print(f"Comparing models with {config.k_min} to {config.k_max} experts...")
        results = compare_models(config=config)

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
