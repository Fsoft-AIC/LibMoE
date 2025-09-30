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
    for j in range(model.num_experts):
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
    for j in range(model.num_experts):
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

    # Compute responsibilities - unpack the tuple returned by e_step
    responsibilities, _ = model.e_step(X, Y)

    # Create a colormap
    cmap = plt.cm.get_cmap('tab10', model.num_experts)

    plt.figure(figsize=(12, 8))

    # Assign each point to the expert with highest responsibility
    expert_assignment = np.argmax(responsibilities, axis=1)

    # Plot data points colored by expert assignment
    for j in range(model.num_experts):
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
    expert_means = np.zeros((len(X), model.num_experts))
    for j in range(model.num_experts):
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


# ==================== VORONOI-STYLE ERROR FOR SHARED SETTINGS ====================
def _pairwise_L1(*diffs):
    """Convenience: L1‑distance for several parameter arrays."""
    return sum(np.abs(d) for d in diffs)


def _pairwise_L2(*diffs):
    """Convenience: squared‑L2 (element‑wise) for several parameter arrays."""
    return sum(d**2 for d in diffs)


def voronoi_D2_linear(
        # ---------- group‑1 (weight‐vector mixture) ----------
        omega_hat, kappa1_hat, kappa0_hat, tau_hat,
        omega_true, kappa1_true, kappa0_true, tau_true,
        # ---------- group‑2 (softmax‐gated mixture) ----------
        beta0_hat, beta1_hat, eta1_hat, eta0_hat, nu_hat,
        beta0_true, beta1_true, eta1_true, eta0_true, nu_true,
        *,
        r1=1.0,   # exponent   r¹(|V₁,j|)
        r2=1.0    # exponent   r²(|V₂,j|)
    ):
    """
    Compute 𝓓₂((G₁,G₂),(G₁*,G₂*)) exactly as in Eq.(7).

    Parameters r1 and r2 let you reproduce the paper’s |V|‑dependent exponents.
    If you set them to 1.0 you obtain the leading‑order L¹ terms; if you set
    them to 0.5 you reproduce the “/2” exponent etc., exactly as written in eq.(7).
    """

    # ─────────────────────────────────────────────────────────────────────
    # GROUP‑1 : assignment of fitted comp i → closest true comp j
    # distance based on (‖Δκ1‖₁ + |Δκ0| + |Δτ|)
    # ─────────────────────────────────────────────────────────────────────
    d1 = _pairwise_L1(
            kappa1_hat[:, None, :] - kappa1_true[None, :, :],
            kappa0_hat[:, None]    - kappa0_true[None, :],
            tau_hat[:, None]       - tau_true[None, :]
         )                                        # shape (k1_hat , k1_true)
    assign1 = np.argmin(d1, axis=1)               # Voronoi cell index for each i
    k1_true = omega_true.shape[0]

    # For bookkeeping, collect cells i ∈ V₁,j
    V1 = [np.where(assign1 == j)[0] for j in range(k1_true)]

    # ─── a) mass‑mismatch term for group‑1 ──────────────────────────────
    mass_err_1 = sum(
        abs(omega_hat[V].sum() - omega_true[j])
        for j, V in enumerate(V1)
    )

    # ─── b) parameter‑mismatch terms for group‑1 ────────────────────────
    param_err_1_L1 = sum(
        (omega_hat[V] *
         _pairwise_L1(
             kappa1_hat[V] - kappa1_true[j],
             kappa0_hat[V] - kappa0_true[j],
             tau_hat[V]    - tau_true[j]
         )).sum()
        for j, V in enumerate(V1) if len(V)
    )

    param_err_1_L2 = sum(
        (omega_hat[V] *
         _pairwise_L2(
             kappa1_hat[V] - kappa1_true[j],
             kappa0_hat[V] - kappa0_true[j],
             tau_hat[V]    - tau_true[j]
         ) ** (r1 / 2.0)       # raise to r¹(|V₁,j|)/2   (paper notation)
         ).sum()
        for j, V in enumerate(V1) if len(V)
    )

    # ─────────────────────────────────────────────────────────────────────
    # GROUP‑2 : assignment of fitted comp i → closest true comp j
    # distance based on (‖Δβ1‖₁ + ‖Δη1‖₁ + |Δη0| + |Δν|)
    # weighting by exp(β0_i)
    # ─────────────────────────────────────────────────────────────────────
    exp_beta0_hat  = np.exp(beta0_hat)
    exp_beta0_true = np.exp(beta0_true)

    d2 = _pairwise_L1(
            beta1_hat[:, None, :] - beta1_true[None, :, :],
            eta1_hat[:, None, :]  - eta1_true[None, :, :],
            eta0_hat[:, None]     - eta0_true[None, :],
            nu_hat[:, None]       - nu_true[None, :]
         )                                        # shape (k2_hat , k2_true)
    assign2 = np.argmin(d2, axis=1)
    k2_true = beta0_true.shape[0]
    V2 = [np.where(assign2 == j)[0] for j in range(k2_true)]

    # ─── a) mass‑mismatch term for group‑2 ──────────────────────────────
    mass_err_2 = sum(
        abs(exp_beta0_hat[V].sum() - exp_beta0_true[j])
        for j, V in enumerate(V2)
    )

    # ─── b) parameter‑mismatch terms for group‑2 ────────────────────────
    param_err_2_L1 = sum(
        (exp_beta0_hat[V] *
         _pairwise_L1(
             beta1_hat[V] - beta1_true[j],
             eta1_hat[V]  - eta1_true[j],
             eta0_hat[V]  - eta0_true[j],
             nu_hat[V]    - nu_true[j]
         )).sum()
        for j, V in enumerate(V2) if len(V)
    )

    param_err_2_L2 = sum(
        (exp_beta0_hat[V] *
         _pairwise_L2(
             beta1_hat[V] - beta1_true[j],
             eta1_hat[V]  - eta1_true[j],
             eta0_hat[V]  - eta0_true[j],
             nu_hat[V]    - nu_true[j]
         ) ** (r2 / 2.0)
        ).sum()
        for j, V in enumerate(V2) if len(V)
    )

    # ─────────────────────────────────────────────────────────────────────
    # Final loss:   Σ mass  + Σ parameter‑mis‑match   (four rows in Eq.(7))
    # ─────────────────────────────────────────────────────────────────────
    D2 = (
        mass_err_1 + mass_err_2 +
        param_err_1_L1 + param_err_1_L2 +
        param_err_2_L1 + param_err_2_L2
    )
    return D2

