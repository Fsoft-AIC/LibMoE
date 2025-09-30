import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# def generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, a_shared_true, 
#                   b_shared_true, sigma_shared_true, omega_shared_true, K, router_type):
#     # Ensure n is an integer
#     n = int(n)

#     # Set a fixed seed for reproducibility
#     rng = default_rng()
#     k_true = len(beta0_true)

#     # Generate X as a 1D array of shape (n,)
#     # Use normal distribution as in the original code
#     X = rng.normal(loc=0.0, scale=1.0, size=n)
#     Y = np.zeros(n)

#     # Reshape X for matrix operations if needed
#     X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X

#     # Compute gating probabilities using topK softmax or sigmoid
#     # First compute logits
#     logits = np.zeros((n, k_true))
#     for i in range(n):
#         logits[i] = beta1_true * X[i] + beta0_true

#     # For each data point, find top-K experts
#     for i in range(n):
#         # Find top-K indices
#         topk_idx = np.argsort(logits[i])[-K:]

#         # Compute softmax or sigmoid over top-K
#         if router_type == "softmax":
#             topk_logits = logits[i, topk_idx]
#             max_logit = np.max(topk_logits)
#             exp_logits = np.exp(topk_logits - max_logit)
#             weights = exp_logits / np.sum(exp_logits)
#         elif router_type == "sigmoid":
#             topk_logits = 1 / (1 + np.exp(-logits[i, topk_idx]))
#             weights = topk_logits / np.sum(topk_logits)
#         else:
#             raise ValueError("Invalid router type. Choose 'softmax' or 'sigmoid'.")
        
#         # Randomly pick one expert from top-K based on weights
#         chosen_expert_local = rng.choice(K, p=weights)
#         chosen_expert = topk_idx[chosen_expert_local]

#         # Generate Y value from the chosen expert and shared expert
#         expert_mean = a_true[chosen_expert] * X[i] + b_true[chosen_expert]
#         shared_mean = np.mean([omega * (a * X[i] + b) for omega, a, b in zip(omega_shared_true, a_shared_true, b_shared_true)])
        
#         # mean = expert_mean + shared_mean
#         expert_stdev = sigma_true[chosen_expert]
#         shared_stdev = np.mean(np.sqrt(omega_shared_true) * sigma_shared_true)
#         # stdev = np.sqrt(expert_stdev**2 + shared_stdev**2)
#         Y[i] = 0.5 * rng.normal(shared_mean, shared_stdev) + 0.5 * rng.normal(expert_mean, expert_stdev)

#     return X, Y


def generate_data(n,
                  beta0_true, beta1_true,
                  a_true, b_true, sigma_true,
                  a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true,
                  K, router_type,
                  dist_type="overlap"):
    """
    Generates (X, Y) with optional shared‑expert structure, in one of three modes:
      - dist_type="overlap": two functions with overlapping x‑ranges
      - dist_type="multitask": piecewise linear / sine / quadratic regions
      - dist_type="denoise": linear + a global high‑freq noise (shared as denoiser)
    """

    rng = default_rng()
    n = int(n)
    k_true = len(beta0_true)

    # --- 1) SAMPLE X & BASE Y from chosen global pattern ---
    if dist_type == "overlap":
        # overlap sine vs. quadratic
        X = rng.uniform(-2, 2, size=n)
        mask = X < 0
        y_base = np.empty(n)
        y_base[mask]   = np.sin(3 * X[mask]) + rng.normal(0, 0.2, mask.sum())
        y_base[~mask]  = (-X[~mask]**2 + 2) + rng.normal(0, 0.3, (~mask).sum())

    elif dist_type == "multitask":
        # piecewise: linear ←, sine centre, quadratic →
        thirds = [n//3, 2*n//3]
        X1 = rng.uniform(-3, -1, size=thirds[0])
        X2 = rng.uniform(-1, 1, size=thirds[1]-thirds[0])
        X3 = rng.uniform(1, 3, size=n - thirds[1])
        X  = np.concatenate([X1, X2, X3])
        y1 = 2 * X1 + rng.normal(0, 0.5, X1.shape)
        y2 = np.sin(5 * X2) + rng.normal(0, 0.2, X2.shape)
        y3 = (-X3**2 + 5) + rng.normal(0, 0.5, X3.shape)
        y_base = np.concatenate([y1, y2, y3])

    elif dist_type == "denoise":
        # simple 2‑piece linear + shared high‑freq noise
        X = rng.uniform(-3, 3, size=n)
        y_lin = np.where(X < 0, 2*X + 1, -3*X + 2)
        noise = np.sin(10 * X) * 0.3
        y_base = y_lin + noise + rng.normal(0, 0.1, size=n)

    else:
        raise ValueError(f"Unknown dist_type={dist_type}")

    # reshape for gating
    X = np.asarray(X)
    Y = np.zeros(n)
    logits = np.outer(X, beta1_true) + beta0_true  # shape (n, k_true)

    # --- 2) ROUTE and SAMPLE from expert + shared mixtures ---
    for i in range(n):
        # top-K
        topk_idx = np.argsort(logits[i])[-K:]
        if router_type == "softmax":
            vals = logits[i, topk_idx]
            exps = np.exp(vals - vals.max())
            weights = exps / exps.sum()
        elif router_type == "sigmoid":
            vals = 1/(1+np.exp(-logits[i, topk_idx]))
            weights = vals / vals.sum()
        else:
            raise ValueError("router_type must be 'softmax' or 'sigmoid'")

        choice = rng.choice(K, p=weights)
        expert = topk_idx[choice]

        # expert prediction
        m_exp = a_true[expert]*X[i] + b_true[expert]
        s_exp = sigma_true[expert]
        # shared expert aggregate
        shared_means = [o*(a*X[i] + b)
                        for o,a,b in zip(omega_shared_true, a_shared_true, b_shared_true)]
        m_sha = np.mean(shared_means)
        s_sha = np.mean(np.sqrt(omega_shared_true)*sigma_shared_true)

        # combine 50/50 mixture
        Y[i] = 0.5 * rng.normal(m_sha, s_sha) + 0.5 * rng.normal(m_exp, s_exp)

    return X, Y


# import numpy as np
# from numpy.random import default_rng

# def generate_data( n,
#                    beta0_true, beta1_true,
#                    a_true, b_true, sigma_true,
#                    a_shared_true, b_shared_true, sigma_shared_true, omega_shared_true,
#                    K, router_type ):
#     rng     = default_rng()
#     n       = int(n)
#     k_true  = len(beta0_true)

#     # 1) sample X and build logits
#     X       = rng.normal(0, 1, size=n)
#     logits  = np.outer(X, beta1_true) + beta0_true
#     Y       = np.zeros(n)

#     for i in range(n):
#         # --- routing ---
#         topk_idx = np.argsort(logits[i])[-K:]
#         vals     = logits[i, topk_idx]
#         if router_type=="softmax":
#             exps    = np.exp(vals - vals.max())
#             weights = exps / exps.sum()
#         else:  # sigmoid
#             sigs    = 1/(1+np.exp(-vals))
#             weights = sigs / sigs.sum()

#         choice        = rng.choice(K, p=weights)
#         expert        = topk_idx[choice]

#         # --- expert + shared means & stdevs ---
#         expert_mean   = a_true[expert]*X[i] + b_true[expert]
#         expert_stdev  = sigma_true[expert]

#         # shared is always active
#         shared_means  = [o*(a*X[i]+b)
#                           for o,a,b in zip(omega_shared_true,
#                                            a_shared_true,
#                                            b_shared_true)]
#         shared_mean   = np.mean(shared_means)
#         shared_stdev  = np.mean(np.sqrt(omega_shared_true)*sigma_shared_true)

#         # --- additive combination ---
#         mean_tot      = expert_mean + shared_mean
#         var_tot       = expert_stdev**2 + shared_stdev**2
#         Y[i]          = rng.normal(loc=mean_tot,
#                                   scale=np.sqrt(var_tot))

#     return X, Y



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
