import numpy as np

def _L1(*diffs):
    """Element‑wise L1 distance summed across supplied arrays."""
    return sum(np.abs(d) for d in diffs)

def _L2sq(*diffs):
    """Element‑wise squared L2 distance summed across supplied arrays."""
    return sum(d**2 for d in diffs)


def voronoi_D1_strong(
        # -------- group‑1 (weights) ------------------------------------
        omega_hat, kappa_hat, tau_hat,
        omega_true, kappa_true, tau_true,
        # -------- group‑2 (softmax‑gated) ------------------------------
        beta0_hat, beta1_hat, eta_hat, nu_hat,
        beta0_true, beta1_true, eta_true, nu_true):
    """
    Voronoi loss 𝓓₁((G₁,G₂),(G₁*,G₂*)) for *strongly identifiable* experts.
    Implements Eq.(5) of the paper.
    """

    # =============== GROUP‑1  (ω, κ, τ) ================================
    d1 = _L1(
            kappa_hat[:, None, :] - kappa_true[None, :, :],
            tau_hat[:, None]      - tau_true[None, :])
    assign1 = np.argmin(d1, axis=1)                 # Voronoi cell index j for each i
    k1_true = omega_true.shape[0]
    V1 = [np.where(assign1 == j)[0] for j in range(k1_true)]

    # ---- mass mismatch ------------------------------------------------
    mass_err_1 = sum(
        abs(omega_hat[V].sum() - omega_true[j])
        for j, V in enumerate(V1)
    )

    # ---- parameter mismatch ( |V| = 1  →  L1 ;  |V| > 1 →  squared L2 )
    param_err_1 = 0.0
    for j, V in enumerate(V1):
        if not len(V):
            continue
        Δκ = kappa_hat[V] - kappa_true[j]
        Δτ = tau_hat[V]   - tau_true[j]
        if len(V) == 1:                                   # |V₁,j| = 1
            param_err_1 += (omega_hat[V] * _L1(Δκ, Δτ)).sum()
        else:                                             # |V₁,j| > 1
            param_err_1 += (omega_hat[V] * _L2sq(Δκ, Δτ)).sum()

    # =============== GROUP‑2  (β₀, β₁, η, ν) ===========================
    exp_beta0_hat  = np.exp(beta0_hat)
    exp_beta0_true = np.exp(beta0_true)

    d2 = _L1(
            beta1_hat[:, None, :] - beta1_true[None, :, :],
            eta_hat[:,  None, :]  - eta_true[None, :, :],
            nu_hat[:,  None]      - nu_true[None, :])
    assign2 = np.argmin(d2, axis=1)
    k2_true = beta0_true.shape[0]
    V2 = [np.where(assign2 == j)[0] for j in range(k2_true)]

    # ---- mass mismatch ------------------------------------------------
    mass_err_2 = sum(
        abs(exp_beta0_hat[V].sum() - exp_beta0_true[j])
        for j, V in enumerate(V2)
    )

    # ---- parameter mismatch ( |V| = 1  →  L1 ;  |V| > 1 →  squared L2 )
    param_err_2 = 0.0
    for j, V in enumerate(V2):
        if not len(V):
            continue
        Δβ1 = beta1_hat[V] - beta1_true[j]
        Δη  = eta_hat[V]  - eta_true[j]
        Δν  = nu_hat[V]   - nu_true[j]
        if len(V) == 1:                                   # |V₂,j| = 1
            param_err_2 += (exp_beta0_hat[V] *
                            _L1(Δβ1, Δη, Δν)).sum()
        else:                                             # |V₂,j| > 1
            param_err_2 += (exp_beta0_hat[V] *
                            _L2sq(Δβ1, Δη, Δν)).sum()

    # =============== TOTAL =============================================
    D1 = mass_err_1 + mass_err_2 + param_err_1 + param_err_2szz 
    return D1
