import numpy as np


def voronoi_style_error(beta0_hat, beta1_hat, a_hat, b_hat, sigma_hat,
                           beta0_true, beta1_true, a_true, b_true, sigma_true):
    # ‑‑ parameter‑distance matrix (scalar parameters shown; extend to vectors if needed)
    d_beta1 = np.abs(beta1_hat[:, np.newaxis] - beta1_true[np.newaxis, :])
    d_a     = np.abs(a_hat[:, np.newaxis]     - a_true[np.newaxis, :])
    d_b     = np.abs(b_hat[:, np.newaxis]     - b_true[np.newaxis, :])
    d_sigma = np.abs(sigma_hat[:, np.newaxis] - sigma_true[np.newaxis, :])
    d_total = d_beta1 + d_a + d_b + d_sigma
    min_dist = np.min(d_total, axis=1)
    
    # parameter mismatch term
    param_err = np.sum(np.exp(beta0_hat) * min_dist)

    # compute mass mismatch term
    mask = (np.argmin(d_total, axis=1) == 1)
    exp_beta0_assigned = np.zeros_like(beta0_hat)
    exp_beta0_assigned[mask] = np.exp(beta0_hat[mask])
    
    exp_beta0_true_assigned = np.zeros_like(beta0_true)
    exp_beta0_true_assigned[mask] = np.exp(beta0_true[mask])
    
    # compute mass mismatch term based on assign position
    mass_err = np.sum(np.abs(exp_beta0_assigned - exp_beta0_true_assigned))

    return param_err + mass_err



    # ─────────────────────────────────────────────────────────────────────
    # GROUP‑2 : assignment of fitted comp i → closest true comp j
    # distance based on (‖Δβ1‖₁ + ‖Δη1‖₁ + |Δη0| + |Δν|)
    # weighting by exp(β0_i)
    d2 = _pairwise_norm(
            beta1_hat[:, np.newaxis] - beta1_true[np.newaxis, :],
            eta1_hat[:, np.newaxis]  - eta1_true[np.newaxis, :]
         ) + _pairwise_L1(
            eta0_hat[:, np.newaxis]     - eta0_true[np.newaxis, :],
            nu_hat[:, np.newaxis]       - nu_true[np.newaxis, :]
         )  # shape (k2_hat , k2_true)
         
    exp_beta0_hat  = np.exp(beta0_hat)
    exp_beta0_true = np.exp(beta0_true)
    
    # ─── a) mass‑mismatch term for group‑2 ──────────────────────────────
    mass_err_2 = sum(
        abs(exp_beta0_hat[j] - exp_beta0_true[j])
        for j in range(len(beta0_hat))
    )
    
    # ─── b) parameter‑mismatch terms for group‑2 ────────────────────────
    param_err_2_L1 = sum(
        (exp_beta0_hat[j] * d2[j])
        for j in range(len(beta0_hat))
    )

    # ─────────────────────────────────────────────────────────────────────
    # Final loss:   Σ mass  + Σ parameter‑mis‑match   (four rows in Eq.(7))
    # ─────────────────────────────────────────────────────────────────────
    D2 = (
        mass_err_1 + mass_err_2 + param_err_1_L1 + param_err_1_L2 + param_err_2_L1 + param_err_2_L2
    )