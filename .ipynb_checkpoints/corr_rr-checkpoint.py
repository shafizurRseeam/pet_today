








import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------

def make_domain_map(df):
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
# p_y optimizer (categorical-safe, matches your Proposition)
# ---------------------------------------------------------------------

def optimal_p_y(f_a, f_b, epsilon, n, domain):
    """
    Closed-form minimizer of average MSE from your Proposition.
    epsilon is unused in this closed-form; kept for API compatibility.
    """
    k = len(domain)
    n2 = float(n)
    num = 0.0
    den = 0.0
    for v in domain:
        fa = float(f_a.get(v, 0.0))
        fb = float(f_b.get(v, 0.0))
        e  = 2.0*fb - 1.0
        d0 = 1.0 - fa - fb
        a0 = fa - fb
        num += (d0*e)/(2.0*k) - (a0*e)/(2.0*n2*k)
        den += (e*e)*((1.0/(4.0*k)) - (1.0/(4.0*n2*k)))
    if abs(den) < 1e-12:
        return 0.5
    return float(np.clip(num/den, 0.0, 1.0))


# --- helpers (add if not present) ---
def _safe_get_freqs(freq_dict, domain):
    return {v: float(freq_dict.get(v, 0.0)) for v in domain}

def build_p_y_table(f_hat_phase1, n2, domain_map):
    """
    Build ordered-pair p_y table using Phase-I debiased marginals.
    Returns: dict[(pivot_col, nonpivot_col)] -> scalar p_y in [0,1]
    """
    cols = list(f_hat_phase1.keys())
    table = {}
    for j in cols:
        for k in cols:
            if j == k:
                continue
            domain_j = domain_map[j]
            domain_k = domain_map[k]
            if len(domain_j) != len(domain_k):
                raise ValueError("Corr-RR assumes a common domain size across attributes.")
            f_a = _safe_get_freqs(f_hat_phase1[j], domain_j)
            f_b = _safe_get_freqs(f_hat_phase1[k], domain_k)
            # epsilon not needed for the closed-form; keep API same
            py  = optimal_p_y(f_a, f_b, epsilon=None, n=n2, domain=domain_j)
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
    combined = {}
    for col in est_A:
        combined[col] = {}
        for v in est_A[col]:
            combined[col][v] = (n1 * est_A[col][v] + n2 * est_B[col][v]) / (n1 + n2)
    return combined

# ---------------------------------------------------------------------
# Optional end-to-end helper
# ---------------------------------------------------------------------

def run_corr_rr(df, epsilon, frac_phase1=0.1, rng=None):
    """
    Convenience pipeline:
      1) Phase I SPL -> f_hat^I
      2) Build p_y table
      3) Phase II synthesize
      4) Debias Phase II as GRR(eps) per column (biased for non-pivots)
      5) Combine Phase I + II by counts
    """
    f_hat_I, df_B, domain_map = corr_rr_phase1_spl(df, epsilon, frac=frac_phase1, rng=rng)
    n1 = len(df) - len(df_B)
    n2 = len(df_B)
    p_y_table = build_p_y_table(f_hat_I, n2=n2, domain_map=domain_map)
    df_II = corr_rr_phase2_perturb(df_B, epsilon, f_hat_I, domain_map, p_y_table, rng=rng)
    est_II = corr_rr_estimate(df_II, domain_map, epsilon)
    combined = combine_phase_estimates(f_hat_I, est_II, n1=n1, n2=n2)
    return combined, p_y_table, (n1, n2)



























# import numpy as np
# import pandas as pd
# from grr import grr_perturb, grr_estimate_frequencies

# # --- p_y optimizer (as you provided) ---
# def optimal_p_y(f_a, f_b, epsilon, n, domain):
#     d = len(domain)
#     exp_eps = np.exp(epsilon)
#     p = exp_eps / (exp_eps + d - 1)
#     q = 1.0 / (exp_eps + d - 1)
#     Δ = p - q

#     S1 = d*(d-1)/2
#     S2 = (d-1)*d*(2*d-1)/6

#     μa  = sum(v * f_a[v] for v in domain)
#     μb  = sum(v * f_b[v] for v in domain)
#     νb2 = sum(v**2 * f_b[v] for v in domain)

#     a0 = μa - μb
#     a1 = 2*μb - S1
#     b1 = 2*νb2 - S2
#     Y0 = (Δ/2)*a0 + (S1/2)

#     α1 = 2 * sum((1 - f_a[v] - f_b[v]) * ((2*f_a[v]-1) + (2*f_b[v]-1)) for v in domain)
#     α2 = sum((2*f_a[v]-1)**2 + (2*f_b[v]-1)**2 for v in domain)

#     num = (b1 - 2*Y0*a1)/(2*n*Δ) + α1/(8*d)
#     den = a1**2/(2*n) - α2/(4*d)

#     p_star = num/den if den != 0 else 1.0
#     return float(np.clip(p_star, 0.0, 1.0))


# # --- Phase I SPL (random sub-sample; unbiased marginals) ---
# def corr_rr_phase1_spl(df, epsilon, frac=0.1):
#     n = len(df)
#     m = max(1, int(round(frac * n)))
#     idx_A = np.random.choice(n, size=m, replace=False)
#     df_A = df.iloc[idx_A]
#     eps_split = epsilon / df.shape[1]

#     reports = []
#     for _, row in df_A.iterrows():
#         reports.append([
#             grr_perturb(row[col], df[col].unique(), eps_split)
#             for col in df.columns
#         ])
#     reports = np.array(reports, dtype=object)

#     est = {}
#     for i, col in enumerate(df.columns):
#         domain = df[col].unique()
#         est[col] = grr_estimate_frequencies(reports[:, i], domain, eps_split)

#     # return the *remaining* users for Phase II
#     df_B = df.drop(index=df.index[idx_A])
#     return est, df_B


# # --- Phase II perturbation (copy from pivot’s *privatized* value) ---
# def corr_rr_phase2_perturb(df, epsilon, f_hat_phase1, domain_map, p_y_table):
#     cols = list(df.columns)
#     d = len(cols)
#     out_rows = []

#     for _, row in df.iterrows():
#         j = np.random.randint(d)
#         pivot_col = cols[j]
#         pivot_domain = domain_map[pivot_col]

#         # PRIVATIZE the pivot once
#         y_pivot = grr_perturb(row[pivot_col], pivot_domain, epsilon)
#         rec = {pivot_col: y_pivot}

#         # Synthesize non-pivots conditional on y_pivot
#         for i, col in enumerate(cols):
#             if i == j:
#                 continue
#             domain = domain_map[col]
#             py = p_y_table.get((pivot_col, col), 0.5)
#             if np.random.rand() < py:
#                 rec[col] = y_pivot
#             else:
#                 others = [v for v in domain if v != y_pivot]
#                 rec[col] = np.random.choice(others) if others else y_pivot

#         out_rows.append(rec)

#     # preserve column order
#     return pd.DataFrame(out_rows, index=df.index)[cols]


# # --- Estimation (column-wise GRR debiasing at ε) ---
# # Note: this treats every column as if passed through GRR(ε).
# # That is biased for synthesized non-pivots, which you already acknowledge.
# def corr_rr_estimate(perturbed_df, domains, epsilon):
#     estimates = {}
#     for col, domain in domains.items():
#         reports = perturbed_df[col].tolist()
#         estimates[col] = grr_estimate_frequencies(reports, domain, epsilon)
#     return estimates


# # --- Combine Phase I (unbiased SPL) + Phase II (biased Corr-RR) by counts ---
# def combine_phase_estimates(est_A, est_B, m, n_minus_m):
#     combined = {}
#     for col in est_A:
#         combined[col] = {}
#         for v in est_A[col]:
#             combined[col][v] = (m * est_A[col][v] + n_minus_m * est_B[col][v]) / (m + n_minus_m)
#     return combined
































