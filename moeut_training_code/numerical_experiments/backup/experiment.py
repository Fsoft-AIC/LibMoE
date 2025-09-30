import numpy as np
import matplotlib.pyplot as plt
import os
from data_generator import generate_data, visualize_data
from model import MoEModel
from trainer import MoETrainer
from utils import (
    plot_expert_predictions,
    plot_gating_probabilities,
    plot_responsibilities,
    evaluate_model
)
from typing import Optional, List, Dict, Any, Union

# Import the ExperimentConfig class for type hints
try:
    from main import ExperimentConfig
except ImportError:
    # Define a placeholder for documentation/type checking
    from dataclasses import dataclass
    @dataclass
    class ExperimentConfig:
        mode: str = 'single'
        k_true: int = 3
        k_fit: int = 3
        k_min: int = 2
        k_max: int = 5
        K: int = 2
        max_iter: int = 100
        seed: int = 42
        verbose: bool = False

def run_experiment(config: Optional[ExperimentConfig] = None, k_true: int = 3, k_fit: int = 3, K: int = 2, max_iter: int = 100, seed: int = 42, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Run an experiment with the MoE model.

    Parameters
    ----------
    config : ExperimentConfig, optional
        Configuration object containing experiment parameters.
        If provided, its values override the individual parameters.
    k_true : int
        True number of experts.
    k_fit : int
        Number of experts to fit.
    K : int
        Number of experts to keep in the gating function.
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    seed : int
        Random seed.
    sample_size : int, optional
        Sample size to run the experiment for.

    Returns
    -------
    results : dict
        Dictionary of results.
    """
    # Use config values if provided, otherwise use individual parameters
    if config is not None:
        k_true = config.k_true
        k_fit = config.k_fit
        K = config.K
        max_iter = config.max_iter
        seed = config.seed
        sample_size = config.sample_sizes

    # Set random seed
    np.random.seed(seed)

    print(f"\nRunning experiment with {sample_size} samples...")

    # get the true parameters from config or generate them if not provided
    if config is not None:
        beta0_true = config.beta0_true
        beta1_true = config.beta1_true
        a_true = config.a_true
        b_true = config.b_true
        sigma_true = config.sigma_true
    else:
        # Generate true parameters
        beta0_true = np.random.randn(k_true)
        beta1_true = np.random.randn(k_true)
        a_true = np.random.randn(k_true)
        b_true = np.random.randn(k_true)
        sigma_true = np.abs(np.random.randn(k_true)) + 0.1

    # Generate data
    # Use init_params for parameter initialization
    X, Y = generate_data(sample_size, beta0_true, beta1_true, a_true, b_true, sigma_true, K)

    # Visualize data
    visualize_data(X, Y)

    # Create and fit model
    model = MoEModel(k=k_fit, K=K)
    trainer = MoETrainer(model, max_iter=max_iter, verbose=True)

    # Pass true parameters for initialization
    true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true)
    trainer.fit(X, Y, true_params=true_params)

    # Plot convergence
    trainer.plot_convergence()
    trainer.plot_parameter_convergence()

    # Plot expert predictions
    plot_expert_predictions(model, X, Y)

    # Plot gating probabilities
    plot_gating_probabilities(model, X)

    # Plot responsibilities
    plot_responsibilities(model, X, Y)

    # Evaluate model
    metrics = evaluate_model(model, X, Y)

    # Print true and estimated parameters
    print("True parameters:")
    print(f"beta0: {beta0_true}")
    print(f"beta1: {beta1_true}")
    print(f"a: {a_true}")
    print(f"b: {b_true}")
    print(f"sigma: {sigma_true}")

    print("\nEstimated parameters:")
    print(f"beta0: {model.beta0}")
    print(f"beta1: {model.beta1}")
    print(f"a: {model.a}")
    print(f"b: {model.b}")
    print(f"sigma: {model.sigma}")

    print("\nEvaluation metrics:")
    print(f"MSE: {metrics['mse']}")
    print(f"Log-likelihood: {metrics['log_likelihood']}")

    return {
        'true_params': {
            'beta0': beta0_true,
            'beta1': beta1_true,
            'a': a_true,
            'b': b_true,
            'sigma': sigma_true
        },
        'estimated_params': {
            'beta0': model.beta0,
            'beta1': model.beta1,
            'a': model.a,
            'b': model.b,
            'sigma': model.sigma
        },
        'metrics': metrics,
        'model': model,
        'trainer': trainer,
        'data': (X, Y)
    }

def compare_models(config: Optional[ExperimentConfig] = None, n: int = 1000, k_true: int = 3, k_fit_values: List[int] = None, K: int = 2, max_iter: int = 100, seed: int = 42) -> Dict[str, Any]:
    """
    Compare models with different numbers of experts.

    Parameters
    ----------
    config : ExperimentConfig, optional
        Configuration object containing experiment parameters.
        If provided, its values override the individual parameters.
    n : int
        Number of data points.
    k_true : int
        True number of experts.
    k_fit_values : list of int
        Numbers of experts to fit.
    K : int
        Number of experts to keep in the gating function.
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary of results.
    """
    # Set default value for k_fit_values
    if k_fit_values is None:
        k_fit_values = [2, 3, 4]

    # Use config values if provided, otherwise use individual parameters
    if config is not None:
        n = config.n
        k_true = config.k_true
        K = config.K
        max_iter = config.max_iter
        seed = config.seed
    # Set random seed
    np.random.seed(seed)

    # Generate true parameters
    beta0_true = np.random.randn(k_true)
    beta1_true = np.random.randn(k_true)
    a_true = np.random.randn(k_true)
    b_true = np.random.randn(k_true)
    sigma_true = np.abs(np.random.randn(k_true)) + 0.1

    # Generate data
    # Use init_params for parameter initialization
    X, Y = generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K)

    # Visualize data
    visualize_data(X, Y)

    results = {}

    for k_fit in k_fit_values:
        print(f"\nFitting model with {k_fit} experts...")

        # Create and fit model
        model = MoEModel(k=k_fit, K=K)
        trainer = MoETrainer(model, max_iter=max_iter, verbose=True)

        # Pass true parameters for initialization
        true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true)
        trainer.fit(X, Y, true_params=true_params)

        # Evaluate model
        metrics = evaluate_model(model, X, Y)

        # Store results
        results[k_fit] = {
            'model': model,
            'trainer': trainer,
            'metrics': metrics
        }

        # Print metrics
        print(f"MSE: {metrics['mse']}")
        print(f"Log-likelihood: {metrics['log_likelihood']}")

    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    # Plot comparison of log-likelihoods
    plt.figure(figsize=(10, 6))
    k_values = list(results.keys())
    log_likelihoods = [results[k]['metrics']['log_likelihood'] for k in k_values]
    plt.plot(k_values, log_likelihoods, 'o-')
    plt.xlabel('Number of Experts')
    plt.ylabel('Log-likelihood')
    plt.title('Comparison of Log-likelihoods')
    plt.grid(True)
    plt.savefig(os.path.join(export_dir, 'model_comparison_ll.pdf'))
    plt.close()

    # Plot comparison of MSEs
    plt.figure(figsize=(10, 6))
    mses = [results[k]['metrics']['mse'] for k in k_values]
    plt.plot(k_values, mses, 'o-')
    plt.xlabel('Number of Experts')
    plt.ylabel('MSE')
    plt.title('Comparison of MSEs')
    plt.grid(True)
    plt.savefig(os.path.join(export_dir, 'model_comparison_mse.pdf'))
    plt.close()

    return {
        'true_params': {
            'beta0': beta0_true,
            'beta1': beta1_true,
            'a': a_true,
            'b': b_true,
            'sigma': sigma_true
        },
        'results': results,
        'data': (X, Y)
    }
