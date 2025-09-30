import numpy as np


# ==================== VORONOI-STYLE ERROR FOR SHARED SETTINGS ====================
def _pairwise_L1(*diffs):
    """Convenience: L1‑distance for several parameter arrays."""
    return sum(np.abs(d) for d in diffs)


def _pairwise_L2(*diffs):
    """Convenience: squared‑L2 (element‑wise) for several parameter arrays."""
    return sum(d**2 for d in diffs)

def _pairwise_norm(*diffs):
    """Convenience: L2-norm for several parameter arrays."""
    return np.sqrt(sum(d**2 for d in diffs))


def voronoi_D2_linear_hehe(
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
    breakpoint()
    d1 = _pairwise_L1(
            kappa1_hat[:, np.newaxis] - kappa1_true[np.newaxis, :],
            kappa0_hat[:, np.newaxis]    - kappa0_true[np.newaxis, :],
            tau_hat[:, np.newaxis]       - tau_true[np.newaxis, :]
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
    # GROUP‑1 : assignment of fitted comp i → closest true comp j (shared experts)
    # distance based on (‖Δκ1‖₁ + |Δκ0| + |Δτ|)
    # ─────────────────────────────────────────────────────────────────────
    d1 = _pairwise_L1(
            kappa1_hat[:, np.newaxis] - kappa1_true[np.newaxis, :],
            kappa0_hat[:, np.newaxis]    - kappa0_true[np.newaxis, :],
            tau_hat[:, np.newaxis]       - tau_true[np.newaxis, :]
         )                                        # shape (k1_hat , k1_true)

    # ─── a) mass‑mismatch term for group‑1 ──────────────────────────────
    mass_err_1 = sum(
        abs(omega_hat[j] - omega_true[j]) for j in range(len(omega_hat))
    )
    
    # ─── b) parameter‑mismatch terms for group‑1 ────────────────────────
    param_err_1_L1 = sum(
        (omega_hat[j] * d1[j])
        for j in range(len(omega_hat))
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # GROUP‑2 : assignment of fitted comp i → closest true comp j
    # distance based on (‖Δβ1‖₁ + ‖Δη1‖₁ + |Δη0| + |Δν|)
    # weighting by exp(β0_i)
    d2 = _pairwise_norm(beta1_hat - beta1_true, eta1_hat - eta1_true) + \
        _pairwise_L1(eta0_hat - eta0_true, nu_hat - nu_true)  # shape (k2_hat , k2_true)
         
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
        mass_err_1 + mass_err_2 + param_err_1_L1 + param_err_2_L1
    )
    # breakpoint()
    return D2

