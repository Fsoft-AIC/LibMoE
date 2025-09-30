import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K):
    """
    Generate (X, Y) from a top-K sparse softmax gating Gaussian MoE.

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
    K : int
        Number of experts to keep in the gating function.

    Returns
    -------
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    """
    rng = default_rng()
    k_true = len(beta0_true)

    # Generate X in [0,1] for simplicity
    X = rng.random(n)
    Y = np.zeros(n)

    # For each x, we compute top-K gating and sample from the selected experts
    for i in range(n):
        # Logits for gating
        logits = beta1_true * X[i] + beta0_true
        # Identify top-K indices
        topk_idx = np.argsort(logits)[-K:]
        # Take the sub-vector of logits for those K experts
        topk_logits = logits[topk_idx]

        # Softmax over those K experts
        w = np.exp(topk_logits - np.max(topk_logits))
        w /= np.sum(w)

        # Randomly pick 1 expert among those K, weighted by w
        chosen_expert_local = rng.choice(K, p=w)
        chosen_expert = topk_idx[chosen_expert_local]

        # Sample Y ~ Normal(a_j * x + b_j, sigma_j^2)
        mean = a_true[chosen_expert]*X[i] + b_true[chosen_expert]
        stdev = sigma_true[chosen_expert]
        Y[i] = rng.normal(mean, stdev)

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
