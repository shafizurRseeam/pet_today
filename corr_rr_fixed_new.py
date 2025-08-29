# corr_rr_fixed.py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------

def make_domain_map(df: pd.DataFrame):
    """
    Stable, sorted domains for each column; ensures a common domain size,
    which Corr-RR (as written) assumes.
    """
    domains = {col: sorted(df[col].unique().tolist()) for col in df.columns}
    sizes = {len(v) for v in domains.values()}
    if len(sizes) != 1:
        raise ValueError("Corr-RR assumes a common domain size k across attributes.")
    return domains


def _safe_get_freqs(freq_dict, domain):
    """Ensure every category in domain appears in freq_dict; fill zeros if missing."""
    return {v: float(freq_dict.get(v, 0.0)) for v in domain}


# ---------------------------------------------------------------------
# GRR interfaces expected to be provided by your project
# (imported lazily inside functions to avoid hard dependency here)
# ---------------------------------------------------------------------

def _grr_perturb(x, domain, epsilon):
    from grr import grr_perturb
    return grr_perturb(x, domain, epsilon)

def _grr_estimate_frequencies(reports, domain, epsilon):
    from grr import grr_estimate_frequencies
    return grr_estimate_frequencies(reports, domain, epsilon)


# ---------------------------------------------------------------------
# p_y optimizer (GENERAL k, exact quadratic minimizer)
#   p_y* = - (Σ_v D1_v) / (2 Σ_v D2_v), clipped to [0,1]
#   where:
#     alpha_v = (1-Δ)/k + (Δ/2)[ d0(v) + 2 f_a(v) ]
#     beta_v  = (Δ/2) e(v)
#     D1_v    = d0(v) e(v)/2 + beta_v (1 - 2 alpha_v) / (n2 Δ^2)
#     D2_v    = e(v)^2/4      - (beta_v)^2 / (n2 Δ^2)
#   with:
#     d0(v) = 1 - f_a(v) - f_b(v),   e(v) = 2 f_b(v) - 1,
#     Δ = p - q,  p = exp(ε)/(exp(ε)+k-1),  q = 1/(exp(ε)+k-1)
# ---------------------------------------------------------------------

def optimal_p_y(f_a, f_b, epsilon, n2, domain):
    """
    Exact closed-form minimizer of the average Phase-II MSE (both attributes),
    for general k-ary GRR. Uses the quadratic coefficients from the paper's
    Proposition (general-k), no binary-only simplifications.

    Args:
      f_a, f_b: dict value->prob for the two attributes' marginals (Phase-I debiased)
      epsilon:  float, Phase-II GRR budget
      n2:       int or float, number of Phase-II users
      domain:   iterable of categories (size k)

    Returns:
      p_y in [0,1]
    """
    k = len(domain)
    n2 = float(n2)

    # Guard: if n2 is tiny, avoid divide-by-zero noise
    if n2 <= 1:
        return 0.5

    # GRR params
    exp_eps = np.exp(float(epsilon))
    p = exp_eps / (exp_eps + k - 1.0)
    q = 1.0     / (exp_eps + k - 1.0)
    Delta = p - q

    # If Δ ~ 0, GRR is maximally noisy; p_y has little effect. Return neutral.
    if abs(Delta) < 1e-12:
        return 0.5

    sum_D1 = 0.0
    sum_D2 = 0.0
    Delta2 = Delta * Delta

    for v in domain:
        fa = float(f_a.get(v, 0.0))
        fb = float(f_b.get(v, 0.0))
        d0 = 1.0 - fa - fb
        e  = 2.0 * fb - 1.0

        alpha = (1.0 - Delta) / k + (Delta / 2.0) * (d0 + 2.0 * fa)
        beta  = (Delta / 2.0) * e

        D1 = (d0 * e) / 2.0 + (beta * (1.0 - 2.0 * alpha)) / (n2 * Delta2)
        D2 = (e * e) / 4.0   - (beta * beta) / (n2 * Delta2)

        sum_D1 += D1
        sum_D2 += D2

    # If denominator ≈ 0, MSE is flat in p_y (no signal) → return neutral 0.5
    if abs(sum_D2) < 1e-12:
        return 0.5

    p_star = - sum_D1 / (2.0 * sum_D2)
    return float(np.clip(p_star, 0.0, 1.0))


def build_p_y_table(f_hat_phase1, n2, domain_map, epsilon):
    """
    Build ordered-pair p_y table using Phase-I debiased marginals.
    Returns: dict[(pivot_col, nonpivot_col)] -> scalar p_y in [0,1]

    NOTE: includes epsilon (needed for general k).
    """
    cols = list(f_hat_phase1.keys())
    table = {}
    # Verify common k
    k_set = {len(domain_map[c]) for c in cols}
    if len(k_set) != 1:
        raise ValueError("Corr-RR assumes a common domain size across attributes.")

    common_domain = domain_map[cols[0]]

    for j in cols:
        for k in cols:
            if j == k:
                continue
            f_a = _safe_get_freqs(f_hat_phase1[j], common_domain)
            f_b = _safe_get_freqs(f_hat_phase1[k], common_domain)
            py  = optimal_p_y(f_a, f_b, epsilon=epsilon, n2=n2, domain=common_domain)
            table[(j, k)] = float(py)
    return table


# ---------------------------------------------------------------------
# Phase I: SPL (private marginals)
# ---------------------------------------------------------------------

def corr_rr_phase1_spl(df, epsilon, frac=0.1, domain_map=None, rng=None):
    """
    Phase I: sample m users (≈ frac*n), perturb all d attributes with GRR(eps/d),
    debias to get unbiased marginals per column.
    Returns:
      f_hat_phase1 (dict: col -> {value->prob}),
      df_B (remaining users for Phase II),
      domain_map
    """
    if rng is None:
        rng = np.random.default_rng()
    if domain_map is None:
        domain_map = make_domain_map(df)

    n = len(df)
    m = max(1, int(round(frac * n)))
    idx_A = rng.choice(n, size=m, replace=False)
    df_A = df.iloc[idx_A]
    eps_split = float(epsilon) / float(df.shape[1])

    cols = list(df.columns)
    reports = []
    for _, row in df_A.iterrows():
        reports.append([
            _grr_perturb(row[col], domain_map[col], eps_split)
            for col in cols
        ])
    reports = np.array(reports, dtype=object)

    f_hat_phase1 = {}
    for j, col in enumerate(cols):
        f_hat_phase1[col] = _grr_estimate_frequencies(reports[:, j], domain_map[col], eps_split)

    df_B = df.drop(index=df.index[idx_A])
    return f_hat_phase1, df_B, domain_map


# ---------------------------------------------------------------------
# Phase II: Corr-aware synthesis
# ---------------------------------------------------------------------

def corr_rr_phase2_perturb(df, epsilon, f_hat_phase1, domain_map, p_y_table, rng=None):
    """
    For each user: pick a pivot uniformly, apply GRR(epsilon) to the pivot,
    synthesize non-pivots by copying the pivot's privatized value with prob p_y,
    else sampling uniformly from remaining k-1 values.
    """
    if rng is None:
        rng = np.random.default_rng()

    cols = list(df.columns)
    d = len(cols)
    out_rows = []

    for _, row in df.iterrows():
        j = rng.integers(d)  # pivot index
        pivot_col = cols[j]
        pivot_domain = domain_map[pivot_col]

        y_pivot = _grr_perturb(row[pivot_col], pivot_domain, epsilon)
        rec = {pivot_col: y_pivot}

        for i, col in enumerate(cols):
            if i == j:
                continue
            key = (pivot_col, col)
            if key not in p_y_table:
                raise KeyError(f"Missing p_y for ordered pair {key}. Build p_y_table first.")
            py = float(p_y_table[key])
            domain = domain_map[col]
            if rng.random() < py:
                rec[col] = y_pivot
            else:
                others = [v for v in domain if v != y_pivot]
                rec[col] = rng.choice(others) if others else y_pivot
        out_rows.append(rec)

    return pd.DataFrame(out_rows, index=df.index)[cols]


# ---------------------------------------------------------------------
# Estimation & combine
# ---------------------------------------------------------------------

def corr_rr_estimate(perturbed_df, domain_map, epsilon):
    """
    Column-wise GRR debiasing at ε for Phase-II reports.
    Unbiased for pivots, generally biased for non-pivots (by design).
    """
    estimates = {}
    for col, domain in domain_map.items():
        reports = perturbed_df[col].tolist()
        estimates[col] = _grr_estimate_frequencies(reports, domain, epsilon)
    return estimates


def combine_phase_estimates(est_A, est_B, n1, n2):
    """
    Count-weighted combine of Phase I (n1 users) and Phase II (n2 users).
    """
    total = float(n1 + n2)
    if total <= 0:
        raise ValueError("n1 + n2 must be positive.")
    combined = {}
    for col in est_A:
        combined[col] = {}
        for v in est_A[col]:
            combined[col][v] = (n1 * est_A[col][v] + n2 * est_B[col][v]) / total
    return combined


# ---------------------------------------------------------------------
# Optional end-to-end helper (Corr-RR only)
# ---------------------------------------------------------------------

def run_corr_rr(df, epsilon, frac_phase1=0.1, rng=None):
    """
    Convenience pipeline:
      1) Phase I SPL -> f_hat^I
      2) Build p_y table
      3) Phase II synthesize
      4) Debias Phase II as GRR(eps) per column (biased for non-pivots)
      5) Combine Phase I + II by counts
    Returns:
      combined_estimates, p_y_table, (n1, n2)
    """
    f_hat_I, df_B, domain_map = corr_rr_phase1_spl(df, epsilon, frac=frac_phase1, rng=rng)
    n1 = len(df) - len(df_B)
    n2 = len(df_B)

    # Build ordered-pair p_y using the general-k optimizer
    p_y_table = build_p_y_table(f_hat_I, n2=n2, domain_map=domain_map, epsilon=epsilon)

    # Phase II synth + debias
    df_II = corr_rr_phase2_perturb(df_B, epsilon, f_hat_I, domain_map, p_y_table, rng=rng)
    est_II = corr_rr_estimate(df_II, domain_map, epsilon)

    # Combine by counts (correct weighting)
    combined = combine_phase_estimates(f_hat_I, est_II, n1=n1, n2=n2)
    return combined, p_y_table, (n1, n2)
