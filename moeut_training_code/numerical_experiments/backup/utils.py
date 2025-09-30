import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def plot_expert_predictions(model, X, Y, filename='expert_predictions.pdf'):
    """
    Plot the predictions of each expert.

    Parameters
    ----------
    model : MoEModel
        The trained model.
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    filename : str
        Name of the file to save the plot to.
    """
    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    # Sort X and Y for better visualization
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    Y_sorted = Y[sort_idx]

    # Create a grid of X values for smooth curves
    X_grid = np.linspace(np.min(X), np.max(X), 1000)

    plt.figure(figsize=(12, 8))

    # Plot data points
    plt.scatter(X_sorted, Y_sorted, alpha=0.3, color='gray', label='Data')

    # Plot expert predictions
    for j in range(model.k):
        mean = model.a[j] * X_grid + model.b[j]
        plt.plot(X_grid, mean, label=f'Expert {j+1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Expert Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(export_dir, filename))
    plt.close()

def plot_gating_probabilities(model, X, filename='gating_probs.pdf'):
    """
    Plot the gating probabilities for each expert.

    Parameters
    ----------
    model : MoEModel
        The trained model.
    X : np.ndarray, shape (n,)
    filename : str
        Name of the file to save the plot to.
    """
    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    # Sort X for better visualization
    X_sorted = np.sort(X)

    # Compute gating probabilities
    gating_probs = model.compute_gating_probs(X_sorted)

    plt.figure(figsize=(12, 8))

    # Plot gating probabilities
    for j in range(model.k):
        plt.plot(X_sorted, gating_probs[:, j], label=f'Expert {j+1}')

    plt.xlabel('X')
    plt.ylabel('Gating Probability')
    plt.title('Gating Probabilities')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(export_dir, filename))
    plt.close()

def plot_responsibilities(model, X, Y, filename='responsibilities.pdf'):
    """
    Plot the responsibilities of each expert for each data point.

    Parameters
    ----------
    model : MoEModel
        The trained model.
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    filename : str
        Name of the file to save the plot to.
    """
    # Ensure the export directory exists
    export_dir = 'export'
    os.makedirs(export_dir, exist_ok=True)

    # Compute responsibilities
    responsibilities = model.e_step(X, Y)

    # Create a colormap
    cmap = plt.cm.get_cmap('tab10', model.k)

    plt.figure(figsize=(12, 8))

    # Assign each point to the expert with highest responsibility
    expert_assignment = np.argmax(responsibilities, axis=1)

    # Plot data points colored by expert assignment
    for j in range(model.k):
        mask = expert_assignment == j
        plt.scatter(X[mask], Y[mask], color=cmap(j), label=f'Expert {j+1}', alpha=0.7)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Points Colored by Expert Assignment')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(export_dir, filename))
    plt.close()

def evaluate_model(model, X, Y):
    """
    Evaluate the model on the data.

    Parameters
    ----------
    model : MoEModel
        The trained model.
    X : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    # Compute gating probabilities
    gating_probs = model.compute_gating_probs(X)

    # Compute expert means
    expert_means = np.zeros((len(X), model.k))
    for j in range(model.k):
        expert_means[:, j] = model.a[j] * X + model.b[j]

    # Compute model predictions
    predictions = np.sum(gating_probs * expert_means, axis=1)

    # Compute MSE
    mse = np.mean((Y - predictions) ** 2)

    # Compute log-likelihood
    likelihoods = model.compute_expert_likelihoods(X, Y)
    weighted_likelihoods = gating_probs * likelihoods
    point_log_likelihoods = np.log(np.sum(weighted_likelihoods, axis=1) + 1e-10)
    log_likelihood = np.sum(point_log_likelihoods)

    return {
        'mse': mse,
        'log_likelihood': log_likelihood
    }

