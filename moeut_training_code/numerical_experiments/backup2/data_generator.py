import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

def generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, b_shared_true, sigma_shared_true, K, router_type):
    """
    Generate (X, Y) from a top-K sparse softmax or sigmoid gating Gaussian MoE.

    Parameters
    ----------
    n : int
        Number of data points.
    beta0_true : array-like of shape (k_true,)
    beta1_true : array-like of shape (k_true,)
        True gating parameters for each expert (1D input -> scalar).
    a_true : array-like of shape (k_true,)
    b_true : array-like of shape (k_true,)
        True expert means are a_i*x + b_i.
    sigma_true : array-like of shape (k_true,)
        True expert stdev for each expert.
    a_shared_true : array-like of shape (num_shared,)
    b_shared_true : array-like of shape (num_shared,)
    K : int
        Number of experts to keep in the gating function.
    router_type : str
        Router type for the gating function.

    Returns
    -------
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    """
    # Ensure n is an integer
    n = int(n)

    # Set a fixed seed for reproducibility
    rng = default_rng()
    k_true = len(beta0_true)

    # Generate X as a 1D array of shape (n,)
    # Use normal distribution as in the original code
    X = rng.normal(loc=0.0, scale=1.0, size=n)
    Y = np.zeros(n)

    # Reshape X for matrix operations if needed
    X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X

    # Compute gating probabilities using topK softmax or sigmoid
    # First compute logits
    logits = np.zeros((n, k_true))
    for i in range(n):
        logits[i] = beta1_true * X[i] + beta0_true

    # For each data point, find top-K experts
    for i in range(n):
        # Find top-K indices
        topk_idx = np.argsort(logits[i])[-K:]

        # Compute softmax or sigmoid over top-K
        if router_type == "softmax":
            topk_logits = logits[i, topk_idx]
            max_logit = np.max(topk_logits)
            exp_logits = np.exp(topk_logits - max_logit)
            weights = exp_logits / np.sum(exp_logits)
        elif router_type == "sigmoid":
            topk_logits = 1 / (1 + np.exp(-logits[i, topk_idx]))
            weights = topk_logits / np.sum(topk_logits)
        else:
            raise ValueError("Invalid router type. Choose 'softmax' or 'sigmoid'.")
        
        # Randomly pick one expert from top-K based on weights
        chosen_expert_local = rng.choice(K, p=weights)
        chosen_expert = topk_idx[chosen_expert_local]

        # Generate Y value from the chosen expert and shared expert
        expert_mean = a_true[chosen_expert] * X[i] + b_true[chosen_expert]
        shared_mean = np.mean([a * X[i] + b for a, b in zip(a_shared_true, b_shared_true)])
        # mean = expert_mean + shared_mean
        expert_stdev = sigma_true[chosen_expert]
        shared_stdev = np.mean(sigma_shared_true)
        # stdev = np.sqrt(expert_stdev**2 + shared_stdev**2)
        Y[i] = rng.normal(shared_mean, shared_stdev) + rng.normal(expert_mean, expert_stdev)

    return X, Y


def visualize_data(X, Y, filename='x_y_relation.pdf'):
    """
    Visualize the relation of X and Y by plotting a scatter plot.

    Parameters
    ----------
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    filename : str
        Name of the file to save the plot to.
    """
    import os

    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X-Y Relationship')
    plt.savefig(os.path.join(export_dir, filename))
    plt.close()
