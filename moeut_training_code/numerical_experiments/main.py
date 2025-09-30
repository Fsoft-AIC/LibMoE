import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 1. DATA GENERATION (top-K gating)
# ----------------------------------------------------------------------
def generate_data(n, beta0_true, beta1_true, a_true, b_true, sigma_true, K):
    """
    Generate (X, Y) samples from a top-K sparse softmax gating Gaussian MoE.
    Exactly matches the ground-truth setup in the paper's experiments.
    PyTorch tensor implementation for GPU acceleration.
    """
    # Convert numpy arrays to PyTorch tensors if they aren't already
    if not isinstance(beta0_true, torch.Tensor):
        beta0_true = torch.tensor(beta0_true, dtype=torch.float32, device=device)
        beta1_true = torch.tensor(beta1_true, dtype=torch.float32, device=device)
        a_true = torch.tensor(a_true, dtype=torch.float32, device=device)
        b_true = torch.tensor(b_true, dtype=torch.float32, device=device)
        sigma_true = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    k_true = len(beta0_true)
    
    # Generate uniform random X values
    X = torch.rand(n, device=device)
    
    # Pre-allocate Y tensor
    Y = torch.zeros(n, device=device)
    
    # Compute all logits at once for all samples
    # X_expanded: [n, 1], beta1_true: [k], result: [n, k]
    X_expanded = X.unsqueeze(1)
    logits = beta1_true * X_expanded + beta0_true
    
    # Process each sample
    for i in range(n):
        # Get top-K indices for this sample
        topk_vals, topk_idx = torch.topk(logits[i], K)
        
        # Compute softmax on top-K logits
        topk_exp = torch.exp(topk_vals - torch.max(topk_vals))
        w = topk_exp / torch.sum(topk_exp)
        
        # Sample from multinomial distribution
        chosen_idx = torch.multinomial(w, 1).item()
        chosen_expert = topk_idx[chosen_idx].item()
        
        # Compute mean and standard deviation
        mean = a_true[chosen_expert] * X[i] + b_true[chosen_expert]
        stdev = sigma_true[chosen_expert]
        
        # Generate normal random value
        Y[i] = mean + stdev * torch.randn(1, device=device)
    
    return X, Y

# ----------------------------------------------------------------------
# 2. GATING + RESPONSIBILITIES
# ----------------------------------------------------------------------
def topk_gating_probs(x, beta0, beta1, K):
    """
    Gating probabilities under top-K gating for a single input x.
    PyTorch tensor implementation.
    """
    k = len(beta0)
    logits = beta1 * x + beta0
    
    # Get top-K indices and values
    topk_vals, topk_idx = torch.topk(logits, K)
    
    # Compute softmax on top-K
    m = torch.max(topk_vals)
    exp_vals = torch.exp(topk_vals - m)
    w = exp_vals / torch.sum(exp_vals)
    
    # Create output tensor with zeros
    probs = torch.zeros_like(beta0)
    probs.index_copy_(0, topk_idx, w)
    
    return probs

def e_step(X, Y, beta0, beta1, a, b, sigma, K):
    """
    E-step: responsibilities r[n,i].
    Vectorized PyTorch implementation.
    """
    n = len(X)
    k = len(beta0)
    r = torch.zeros((n, k), device=device)
    
    # Process each sample
    for i in range(n):
        gating_probs = topk_gating_probs(X[i], beta0, beta1, K)
        
        # Compute means for all experts
        means = a * X[i] + b
        
        # Compute PDF values for all experts
        z_scores = (Y[i] - means) / sigma
        log_pdf = -0.5 * torch.log(2.0 * torch.tensor(np.pi, device=device)) - torch.log(sigma) - 0.5 * z_scores**2
        pdf_vals = torch.exp(log_pdf)
        
        # Compute responsibilities
        numer = gating_probs * pdf_vals
        denom = torch.sum(numer)
        
        # Handle numerical stability
        if denom < 1e-14:
            numer = numer + 1e-14
            denom = torch.sum(numer)
        
        r[i, :] = numer / denom
    
    return r

# ----------------------------------------------------------------------
# 3. M-STEP FOR EXPERT PARAMETERS
# ----------------------------------------------------------------------
def m_step_experts(X, Y, r, a, b, sigma):
    """
    Closed-form weighted least squares for (a[i], b[i]) and MLE for sigma[i].
    PyTorch tensor implementation.
    """
    n, k = r.shape
    
    # Create copies to update
    a_new = a.clone()
    b_new = b.clone()
    sigma_new = sigma.clone()
    
    for i in range(k):
        w_sum = torch.sum(r[:, i])
        if w_sum < 1e-14:
            continue
        
        # Weighted sums
        R = w_sum
        SX = torch.sum(X * r[:, i])
        SY = torch.sum(Y * r[:, i])
        SXX = torch.sum(X**2 * r[:, i])
        SXY = torch.sum(X * Y * r[:, i])
        
        denom = R * SXX - SX * SX
        if torch.abs(denom) < 1e-14:
            # Fallback: set a[i] = 0
            a_new[i] = 0.0
            b_new[i] = SY / R
        else:
            a_new[i] = (R * SXY - SX * SY) / denom
            b_new[i] = (SY - a_new[i] * SX) / R
        
        # Update sigma
        resid = Y - (a_new[i] * X + b_new[i])
        sigma_sq = torch.sum(r[:, i] * (resid**2)) / R
        sigma_new[i] = torch.max(torch.sqrt(sigma_sq), torch.tensor(1e-4, device=device))
    
    return a_new, b_new, sigma_new

# ----------------------------------------------------------------------
# 4. M-STEP FOR GATING
# ----------------------------------------------------------------------
def m_step_gating(X, Y, r, beta0, beta1, a, b, sigma, K, lr=5e-3, steps=10):
    """
    Gating parameter update via gradient-based approach.
    PyTorch tensor implementation.
    """
    k = len(beta0)
    n = len(X)
    b0_new = beta0.clone()
    b1_new = beta1.clone()
    
    for _ in range(steps):
        grad_b0 = torch.zeros_like(beta0)
        grad_b1 = torch.zeros_like(beta1)
        
        for i in range(n):
            gp = topk_gating_probs(X[i], b0_new, b1_new, K)
            diff = r[i, :] - gp
            grad_b0 += diff
            grad_b1 += diff * X[i]
        
        # Update parameters
        b0_new += lr * grad_b0
        b1_new += lr * grad_b1
    
    return b0_new, b1_new

# ----------------------------------------------------------------------
# 5. EM WRAPPER
# ----------------------------------------------------------------------
def fit_topk_moe(X, Y, k, K, true_params=None, max_iter=200, tol=1e-6, lr=5e-3, steps=10):
    """
    Fit a top-K gating MoE via an EM-like approach.
    PyTorch tensor implementation.
    """
    # Initialize parameters
    if true_params is not None:
        # Use the init_params function with true parameters
        beta1, beta0, a, b, sigma = init_params(true_params, len(X))
    else:
        # Fallback to random initialization
        beta0 = torch.normal(mean=0.0, std=1.0, size=(k,), device=device)
        beta1 = torch.normal(mean=0.0, std=1.0, size=(k,), device=device)
        a = torch.normal(mean=0.0, std=1.0, size=(k,), device=device)
        b = torch.normal(mean=0.0, std=1.0, size=(k,), device=device)
        sigma = torch.ones(k, device=device) * 0.5
    
    prev_ll = torch.tensor(-1e15, device=device)
    
    for it in range(max_iter):
        # E-step
        r = e_step(X, Y, beta0, beta1, a, b, sigma, K)
        
        # M-step for experts
        a, b, sigma = m_step_experts(X, Y, r, a, b, sigma)
        
        # M-step for gating
        beta0, beta1 = m_step_gating(X, Y, r, beta0, beta1, a, b, sigma, K, lr=lr, steps=steps)
        
        # Check log-likelihood
        ll = torch.tensor(0.0, device=device)
        for i in range(len(X)):
            gp = topk_gating_probs(X[i], beta0, beta1, K)
            means = a * X[i] + b
            z_scores = (Y[i] - means) / sigma
            log_pdf = -0.5 * torch.log(2.0 * torch.tensor(np.pi, device=device)) - torch.log(sigma) - 0.5 * z_scores**2
            pdf_vals = torch.exp(log_pdf)
            ll += torch.log(torch.sum(gp * pdf_vals) + 1e-14)
        
        # Check convergence
        if torch.abs(ll - prev_ll) < tol:
            break
        
        prev_ll = ll
    
    return beta0, beta1, a, b, sigma

# ----------------------------------------------------------------------
# 6. VORONOI-STYLE ERROR (SIMPLIFIED)
# ----------------------------------------------------------------------
def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                       beta0_true, beta1_true, a_true, b_true, sigma_true):
    """
    A simplified 'Voronoi-style' metric.
    PyTorch tensor implementation.
    """
    # Ensure all inputs are PyTorch tensors
    if not isinstance(beta0_true, torch.Tensor):
        beta0_true = torch.tensor(beta0_true, dtype=torch.float32, device=device)
        beta1_true = torch.tensor(beta1_true, dtype=torch.float32, device=device)
        a_true = torch.tensor(a_true, dtype=torch.float32, device=device)
        b_true = torch.tensor(b_true, dtype=torch.float32, device=device)
        sigma_true = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    k_fit = len(beta0_hat)
    k_true = len(beta0_true)
    
    total_err = torch.tensor(0.0, device=device)
    
    for i in range(k_fit):
        # Initialize with a large value
        best_dist = torch.tensor(1e15, device=device)
        best_j = 0
        
        # Find closest true component
        for j in range(k_true):
            d = (torch.abs(beta1_hat[i] - beta1_true[j]) +
                 torch.abs(a_hat[i] - a_true[j]) +
                 torch.abs(b_hat[i] - b_true[j]) +
                 torch.abs(sigma_hat[i] - sigma_true[j]))
            
            if d < best_dist:
                best_dist = d
                best_j = j
        
        # Weight by exp(beta0_hat[i])
        w_i = torch.exp(beta0_hat[i])
        total_err += w_i * best_dist
    
    return total_err.item()  # Convert to Python scalar for compatibility

# ----------------------------------------------------------------------
# 7. PARAMETER INITIALIZATION
# ----------------------------------------------------------------------
def init_params(true_params, n_samples):
    """
    Starting values for EM algorithm.
    PyTorch tensor implementation.
    """
    n_components = 2
    n_features = 1
    
    # Unpack true parameters
    beta1_true, beta0_true, a_true, b_true, sigma_true = true_params
    
    # Convert to PyTorch tensors if they aren't already
    if not isinstance(beta0_true, torch.Tensor):
        beta0_true = torch.tensor(beta0_true, dtype=torch.float32, device=device)
        beta1_true = torch.tensor(beta1_true, dtype=torch.float32, device=device)
        a_true = torch.tensor(a_true, dtype=torch.float32, device=device)
        b_true = torch.tensor(b_true, dtype=torch.float32, device=device)
        sigma_true = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    # Initialize parameter tensors
    beta1_init = torch.zeros((n_components, n_features), device=device)
    beta0_init = torch.zeros(n_components, device=device)
    a_init = torch.zeros((n_components, n_features), device=device)
    b_init = torch.zeros(n_components, device=device)
    sigma_init = torch.zeros(n_components, device=device)
    
    inds = torch.arange(n_components, device=device)
    
    # Make a partition of starting values near the true components
    while True:
        s_inds = torch.randint(0, n_components, (n_components,), device=device)
        unique_vals = torch.unique(s_inds)
        
        if len(unique_vals) == n_components:
            break
    
    noise_scale = 0.005 * (n_samples ** (-0.083))
    noise_scale_sigma = 0.0005 * (n_samples ** (-0.25))
    
    for k in range(n_components):
        idx = s_inds[k].item()
        
        # Add noise to initial values
        beta1_init[k] = beta1_true[idx] + torch.normal(0, noise_scale, size=(n_features,), device=device)
        beta0_init[k] = beta0_true[idx] + torch.normal(0, noise_scale, size=(1,), device=device)
        a_init[k] = a_true[idx] + torch.normal(0, noise_scale, size=(n_features,), device=device)
        b_init[k] = b_true[idx] + torch.normal(0, noise_scale, size=(1,), device=device)
        sigma_init[k] = sigma_true[idx] + torch.abs(torch.normal(0, noise_scale_sigma, size=(1,), device=device))
    
    return (beta1_init.flatten(), beta0_init, a_init.flatten(), b_init, sigma_init)

# ----------------------------------------------------------------------
# 8. MAIN: REPRODUCE FIGURE 3-LIKE EXPERIMENT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Ground-truth parameters
    beta0_true = np.array([-8.0, 0.0])
    beta1_true = np.array([25.0, 0.0])
    a_true = np.array([-20.0, 20.0])
    b_true = np.array([15.0, -5.0])
    sigma_true = np.array([0.3, 0.4])
    
    # Convert to PyTorch tensors
    beta0_true_tensor = torch.tensor(beta0_true, dtype=torch.float32, device=device)
    beta1_true_tensor = torch.tensor(beta1_true, dtype=torch.float32, device=device)
    a_true_tensor = torch.tensor(a_true, dtype=torch.float32, device=device)
    b_true_tensor = torch.tensor(b_true, dtype=torch.float32, device=device)
    sigma_true_tensor = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    # For the ground-truth gating, we do top-1 gating
    K_true = 1
    
    # Scenarios to test
    scenarios = [
        ("Exact-Specified", 2, 1)
    ]
    
    # Sample sizes to test
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    # Number of trials for each sample size
    n_trials = 40
    
    # Create figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for ax_idx, (title, k_fit, K_fit) in enumerate(scenarios):
        avg_errors = []
        std_errors = []
        print(f"\nRunning {title} scenario...")
        
        for n in tqdm(sample_sizes):
            errors = []
            for _ in range(n_trials):
                # Data generation from the *true* top-1 gating with 2 experts
                X, Y = generate_data(n, beta0_true_tensor, beta1_true_tensor, 
                                    a_true_tensor, b_true_tensor, sigma_true_tensor, K_true)
                
                # Fit the top-K gating MoE
                true_params = (beta1_true_tensor, beta0_true_tensor, a_true_tensor, 
                              b_true_tensor, sigma_true_tensor)
                beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat = fit_topk_moe(
                    X, Y, k=k_fit, K=K_fit, true_params=true_params,
                    max_iter=2000, tol=1e-6, lr=5e-3, steps=10
                )
                
                # Compute "Voronoi-style" error
                err = voronoi_style_error(
                    beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                    beta0_true_tensor, beta1_true_tensor, a_true_tensor, 
                    b_true_tensor, sigma_true_tensor
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
    plt.savefig("gpt_code/export/voronoi_style_error_pytorch.pdf")
    print("Experiment completed. Results saved to gpt_code/export/voronoi_style_error_pytorch.pdf")
