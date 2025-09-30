import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# ----------------------------------------------------------------------
# 1. DATA GENERATION (top-K gating)
# ----------------------------------------------------------------------
def generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K):
    """
    Generate (X, Y) samples from a top-K sparse softmax gating Gaussian MoE.
    Exactly matches the ground-truth setup in the paper's experiments.
    """
    rng = default_rng()
    k_true = len(beta0_true)

    # *** CHANGED: to more closely match the paper, we can draw X from Uniform(0,1)
    X = rng.random(n)

    Y = np.zeros(n)
    for i in range(n):
        logits = beta1_true * X[i] + beta0_true
        topk_idx = np.argsort(logits)[-K:]  # top-K
        topk_logits = logits[topk_idx]
        # Softmax restricted to top-K
        topk_exp = np.exp(topk_logits - np.max(topk_logits))
        w = topk_exp / np.sum(topk_exp)
        chosen_expert = rng.choice(topk_idx, p=w)
        # Gaussian draw
        mean = a_true[chosen_expert]*X[i] + b_true[chosen_expert]
        stdev = sigma_true[chosen_expert]
        Y[i] = rng.normal(mean, stdev)

    return X, Y

# ----------------------------------------------------------------------
# 2. GATING + RESPONSIBILITIES
# ----------------------------------------------------------------------
def topk_gating_probs(x, beta0, beta1, K):
    """
    Gating probabilities under top-K gating for a single input x.
    """
    k = len(beta0)
    logits = beta1*x + beta0
    # Identify top-K
    topk_idx = np.argsort(logits)[-K:]
    # Softmax on those
    topk_logits = logits[topk_idx]
    m = np.max(topk_logits)
    exp_vals = np.exp(topk_logits - m)
    w = exp_vals / np.sum(exp_vals)
    probs = np.zeros(k)
    probs[topk_idx] = w
    return probs

def e_step(X, Y, beta0, beta1, a, b, sigma, K):
    """
    E-step: responsibilities r[n,i].
    """
    n = len(X)
    k = len(beta0)
    r = np.zeros((n, k))
    for i in range(n):
        gating_probs = topk_gating_probs(X[i], beta0, beta1, K)
        means = a*X[i] + b
        pdf_vals = (1.0/np.sqrt(2.0*np.pi*sigma**2)) * np.exp(-0.5*((Y[i]-means)/sigma)**2)
        numer = gating_probs * pdf_vals
        denom = np.sum(numer)
        if denom < 1e-14:
            numer += 1e-14
            denom = np.sum(numer)
        r[i, :] = numer / denom
    return r

# ----------------------------------------------------------------------
# 3. M-STEP FOR EXPERT PARAMETERS
# ----------------------------------------------------------------------
def m_step_experts(X, Y, r, a, b, sigma):
    """
    Closed-form weighted least squares for (a[i], b[i]) and MLE for sigma[i].
    """
    n, k = r.shape
    for i in range(k):
        w_sum = np.sum(r[:, i])
        if w_sum < 1e-14:
            continue
        # Weighted sums
        R = w_sum
        SX = np.sum(X*r[:, i])
        SY = np.sum(Y*r[:, i])
        SXX = np.sum(X**2*r[:, i])
        SXY = np.sum(X*Y*r[:, i])
        denom = R*SXX - SX*SX
        if abs(denom) < 1e-14:
            # fallback: set a[i] = 0
            a[i] = 0.0
            b[i] = SY / R
        else:
            a[i] = (R*SXY - SX*SY)/denom
            b[i] = (SY - a[i]*SX)/R
        # sigma
        resid = Y - (a[i]*X + b[i])
        sigma_sq = np.sum(r[:, i]*(resid**2))/R
        sigma[i] = max(np.sqrt(sigma_sq), 1e-4)
    return a, b, sigma

# ----------------------------------------------------------------------
# 4. M-STEP FOR GATING
# ----------------------------------------------------------------------
def m_step_gating(X, Y, r, beta0, beta1, a, b, sigma, K, lr=5e-3, steps=10):
    """
    Gating parameter update via gradient-based approach.
    """
    k = len(beta0)
    n = len(X)
    b0_new = beta0.copy()
    b1_new = beta1.copy()
    for _ in range(steps):
        grad_b0 = np.zeros(k)
        grad_b1 = np.zeros(k)
        for i in range(n):
            gp = topk_gating_probs(X[i], b0_new, b1_new, K)  # gating probs
            for j in range(k):
                diff = (r[i, j] - gp[j])
                grad_b0[j] += diff
                grad_b1[j] += diff*X[i]
        # Update
        b0_new += lr*grad_b0
        b1_new += lr*grad_b1
    return b0_new, b1_new

# ----------------------------------------------------------------------
# 5. EM WRAPPER
# ----------------------------------------------------------------------
def fit_topk_moe(X, Y, k, K, true_params=None, max_iter=200, tol=1e-6, lr=5e-3, steps=10):
    """
    Fit a top-K gating MoE via an EM-like approach.
    """
    # Initialize parameters
    if true_params is not None:
        # Use the init_params function with true parameters
        beta1, beta0, a, b, sigma = init_params(true_params, len(X))
    else:
        # Fallback to random initialization if true parameters are not provided
        rng = default_rng()
        beta0 = rng.normal(loc=0, scale=1, size=k)
        beta1 = rng.normal(loc=0, scale=1, size=k)
        a = rng.normal(loc=0, scale=1, size=k)
        b = rng.normal(loc=0, scale=1, size=k)
        sigma = np.ones(k)*0.5

    prev_ll = -1e15
    for it in range(max_iter):
        r = e_step(X, Y, beta0, beta1, a, b, sigma, K)
        a, b, sigma = m_step_experts(X, Y, r, a, b, sigma)
        beta0, beta1 = m_step_gating(X, Y, r, beta0, beta1, a, b, sigma, K, lr=lr, steps=steps)
        # Check log-likelihood
        ll = 0.0
        for i in range(len(X)):
            gp = topk_gating_probs(X[i], beta0, beta1, K)
            means = a*X[i] + b
            pdf_vals = (1.0/np.sqrt(2.0*np.pi*sigma**2))*np.exp(-0.5*((Y[i]-means)/sigma)**2)
            ll += np.log(np.sum(gp*pdf_vals) + 1e-14)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return beta0, beta1, a, b, sigma

# ----------------------------------------------------------------------
# 6. VORONOI-STYLE ERROR (SIMPLIFIED)
# ----------------------------------------------------------------------
def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                       beta0_true, beta1_true, a_true, b_true, sigma_true):
    """
    A simplified 'Voronoi-style' metric. We do:
      1) For each fitted component i, find the ground-truth j that minimizes
         param-distance (like a Voronoi cell).
      2) Sum up (exp(beta0_hat[i])) * [|beta1_hat[i]-beta1_true[j]| + etc... ].
    This is not the paper's exact D1 or D2, but demonstrates the general shape
    of the metric used in Figure 3.
    """
    k_fit = len(beta0_hat)
    k_true = len(beta0_true)

    total_err = 0.0
    for i in range(k_fit):
        # pick j that is closest in parameter space
        best_j = 0
        best_dist = 1e15
        for j in range(k_true):
            d = (abs(beta1_hat[i] - beta1_true[j]) +
                 abs(a_hat[i] - a_true[j]) +
                 abs(b_hat[i] - b_true[j]) +
                 abs(sigma_hat[i] - sigma_true[j]))
            if d < best_dist:
                best_dist = d
                best_j = j
        # Weighted by exp(beta0_hat[i])
        w_i = np.exp(beta0_hat[i])
        total_err += w_i * best_dist
    return total_err

# ----------------------------------------------------------------------
# 7. PARAMETER INITIALIZATION
# ----------------------------------------------------------------------
def init_params(true_params, n_samples):
    """ Starting values for EM algorithm. """
    # n_samples = 1000
    n_components = 2
    n_features = 1


    (beta1_true, beta0_true, a_true, b_true, sigma_true) = true_params

    beta1_init = np.zeros((n_components, n_features))
    beta0_init = np.zeros(n_components)
    a_init = np.zeros((n_components, n_features))
    b_init = np.zeros(n_components)
    sigma_init = np.zeros(n_components)

    inds = range(n_components)

    # Make a partition of starting values near the true components.
    while True:
        s_inds = np.random.choice(inds, size=n_components)
        unique,counts = np.unique(s_inds, return_counts=True)

        if unique.size == n_components:
            break

    for k in range(n_components):
        beta1_init[k] = (beta1_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083), size=(n_features)))
        beta0_init[k] = (beta0_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083)))
        a_init[k] = a_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083), size=(n_features))
        b_init[k] = (b_true[s_inds[k]] + np.random.normal(0, 0.005*n_samples**(-0.083)))
        sigma_init[k] = sigma_true[s_inds[k]] + np.abs(np.random.normal(0, 0.0005*n_samples**(-0.25)))

    return (beta1_init.flatten(), beta0_init, a_init.flatten(), b_init, sigma_init)

# ----------------------------------------------------------------------
# 8. MAIN: REPRODUCE FIGURE 3-LIKE EXPERIMENT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # *** CHANGED: These are the ground-truth parameters used in the paper's experiments
    # EXACT same as the snippet in the text, but repeated here with comment
    beta0_true = np.array([-8.0, 0.0])
    beta1_true = np.array([25.0, 0.0])
    a_true     = np.array([-20.0, 20.0])
    b_true     = np.array([15.0, -5.0])
    sigma_true = np.array([0.3, 0.4])

    # For the ground-truth gating, we do top-1 gating
    K_true = 1

    # *** CHANGED: We'll replicate two scenarios:
    #     1) EXACT-SPECIFIED: (k=2, K=1)
    #     2) OVER-SPECIFIED: (k=3, K=2)
    scenarios = [
        ("Exact-Specified", 2, 1)
    ]

    # *** CHANGED: More sample sizes, up to 1e5, in log steps
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    # sample_sizes = [1, 10, 20, 50, 100]

    # *** CHANGED: Multiple trials to produce mean+std error bars
    n_trials = 40

    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    for ax_idx, (title, k_fit, K_fit) in enumerate(scenarios):
        avg_errors = []
        std_errors = []
        print(f"\nRunning {title} scenario...")
        from tqdm import tqdm

        for n in tqdm(sample_sizes):
            errors = []
            for _ in range(n_trials):
                # Data generation from the *true* top-1 gating with 2 experts
                X, Y = generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K_true)

                # Fit the top-K gating MoE
                true_params = (beta1_true, beta0_true, a_true, b_true, sigma_true)
                beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat = fit_topk_moe(
                    X, Y, k=k_fit, K=K_fit, true_params=true_params,
                    max_iter=2000, tol=1e-6, lr=5e-3, steps=10
                )

                # Compute "Voronoi-style" error
                err = voronoi_style_error(
                    beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                    beta0_true, beta1_true, a_true, b_true, sigma_true
                )
                errors.append(err)
            avg_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))

        # Plot in log-log
        axes[ax_idx].errorbar(sample_sizes, avg_errors, yerr=std_errors, fmt='-o')
        axes[ax_idx].set_xscale('log')
        axes[ax_idx].set_yscale('log')
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xlabel("n (log scale)")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Voronoi-Style Error (log scale)")

    plt.tight_layout()
    # plt.show()
    plt.savefig("voronoi_style_error.pdf")
