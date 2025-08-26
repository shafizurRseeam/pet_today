# import numpy as np
# import pandas as pd
# from grr import grr_perturb, grr_estimate_frequencies
# from data_utils import correlated_data_generator, get_true_frequencies  # NEW

# # --- p_y Optimization (unchanged) ---
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

#     p_star = num/den if den != 0 else 1
#     return float(np.clip(p_star, 0.0, 1.0))

# # --- Phase 1 SPL (unchanged) ---
# def corr_rr_phase1_spl(df, epsilon, frac=0.1):
#     n = len(df)
#     m = int(frac * n)
#     df_A = df.iloc[:m]
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
#     return est, df.iloc[m:]

# # --- Phase 2 + Estimation (unchanged) ---
# def corr_rr_phase2_perturb(df, epsilon, f_hat_phase1, domain_map, p_y_table):
#     d = len(df.columns)
#     privatized = []
#     for _, row in df.iterrows():
#         j = np.random.randint(d)
#         perturbed = {}
#         for i, col in enumerate(df.columns):
#             domain = domain_map[col]
#             if i == j:
#                 perturbed[col] = grr_perturb(row[col], domain, epsilon)
#             else:
#                 pair = (df.columns[j], col)
#                 py = p_y_table.get(pair, 0.5)
#                 if np.random.rand() < py:
#                     perturbed[col] = row[col]
#                 else:
#                     perturbed[col] = np.random.choice([v for v in domain if v != row[col]])
#         privatized.append(perturbed)
#     return pd.DataFrame(privatized, index=df.index)

# def corr_rr_estimate(perturbed_df, domains, epsilon):
#     estimates = {}
#     n, d = perturbed_df.shape
#     for col, domain in domains.items():
#         reports = perturbed_df[col].tolist()
#         estimates[col] = grr_estimate_frequencies(reports, domain, epsilon)
#     return estimates

# def combine_phase_estimates(est_A, est_B, m, n_minus_m):
#     combined = {}
#     for col in est_A:
#         combined[col] = {}
#         for v in est_A[col]:
#             combined[col][v] = (m * est_A[col][v] + n_minus_m * est_B[col][v]) / (m + n_minus_m)
#     return combined

# corr_rr.py
import numpy as np
import pandas as pd
from grr import grr_perturb, grr_estimate_frequencies


# -------------------- Utilities --------------------

def _stable_domains(df, domains=None):
    """Return {col: sorted unique values} if domains not provided."""
    if domains is not None:
        return domains
    return {c: sorted(pd.unique(df[c]).tolist()) for c in df.columns}

def _numeric_index_moments(f_map, domain):
    """
    Compute moments using numeric indices 0..k-1 over the provided domain order,
    avoiding reliance on actual label values being numeric.
    """
    mu = 0.0
    mu2 = 0.0
    for idx, v in enumerate(domain):
        p = float(f_map.get(v, 0.0))
        mu += idx * p
        mu2 += (idx ** 2) * p
    return mu, mu2


# -------------------- p_y optimization --------------------

def optimal_p_y(f_a, f_b, epsilon, n2, domain):
    """
    Minimize the (approx.) Phase-II MSE for pair (a -> b) to pick p_y in [0,1].
    Uses index-based moments over the given domain order.

    Args:
      f_a, f_b: dict {value: prob} Phase-I marginals for attributes a and b
      epsilon:   Phase-II GRR privacy parameter (pivot attribute is GRR(ε))
      n2:        number of Phase-II users (appears in the approximation)
      domain:    list of category labels (shared order across columns)

    Returns:
      p_y in [0, 1]
    """
    k = len(domain)
    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + k - 1)
    q = 1.0 / (exp_eps + k - 1)
    Delta = p - q

    # Index-based sums over domain 0..k-1
    # S1 = sum i, S2 = sum i^2
    S1 = k * (k - 1) / 2.0
    S2 = (k - 1) * k * (2 * k - 1) / 6.0

    mu_a, _      = _numeric_index_moments(f_a, domain)
    mu_b, mu2_b  = _numeric_index_moments(f_b, domain)

    a0 = mu_a - mu_b
    a1 = 2.0 * mu_b - S1
    b1 = 2.0 * mu2_b - S2
    Y0 = (Delta / 2.0) * a0 + (S1 / 2.0)

    # Terms depending only on probabilities (no index)
    alpha1 = 0.0
    alpha2 = 0.0
    for v in domain:
        fa = float(f_a.get(v, 0.0))
        fb = float(f_b.get(v, 0.0))
        d0 = 1.0 - fa - fb
        ea = 2.0 * fa - 1.0
        eb = 2.0 * fb - 1.0
        alpha1 += (d0) * (ea + eb) * 2.0
        alpha2 += (ea ** 2 + eb ** 2)

    num = (b1 - 2.0 * Y0 * a1) / (2.0 * n2 * Delta) + alpha1 / (8.0 * k)
    den = (a1 ** 2) / (2.0 * n2) - alpha2 / (4.0 * k)

    if np.isclose(den, 0.0):
        return 1.0
    return float(np.clip(num / den, 0.0, 1.0))


def build_p_y_table(est_I, epsilon, n2, domain):
    """
    Convenience: build p_y for every ordered pair (a -> b) from Phase-I marginals.

    Args:
      est_I:   dict {col: {val: prob}} Phase-I marginals
      epsilon: Phase-II GRR budget
      n2:      number of Phase-II users
      domain:  common domain list

    Returns:
      dict { (a_col, b_col): p_y }
    """
    cols = list(est_I.keys())
    return {
        (a, b): optimal_p_y(est_I[a], est_I[b], epsilon, n2, domain)
        for a in cols for b in cols if a != b
    }


# -------------------- Phase I: SPL --------------------

def corr_rr_phase1_spl(df, epsilon, frac=0.1, domains=None):
    """
    Phase I:
      - Take first n1=int(frac*n) users, each perturbs all d attributes with GRR(ε/d).
      - Debias column-wise to get marginals.

    Returns:
      est_I:  {col: {val: prob}}
      df_B:   remaining raw df for Phase II
      domains: stable {col: domain_list}
    """
    n = len(df)
    n1 = max(1, int(frac * n))
    df_A = df.iloc[:n1]
    df_B = df.iloc[n1:]
    d = df.shape[1]
    eps_split = epsilon / d

    domains = _stable_domains(df, domains)
    cols = list(df.columns)

    # Client-side perturbation for Phase I
    reports = []
    for _, row in df_A.iterrows():
        reports.append([
            grr_perturb(row[col], domains[col], eps_split)
            for col in cols
        ])
    reports = np.array(reports, dtype=object)

    # Server-side debiasing (unbiased per column)
    est_I = {}
    for j, col in enumerate(cols):
        est_I[col] = grr_estimate_frequencies(reports[:, j], domains[col], eps_split)

    return est_I, df_B, domains


# -------------------- Phase II: correlation-aware perturbation --------------------

def corr_rr_phase2_perturb(df_B, epsilon, domains, p_y_table):
    """
    Phase II client generation:
      - pick a pivot attribute uniformly,
      - perturb pivot with GRR(ε),
      - for every non-pivot attribute k: with prob p_y[(pivot->k)] copy the pivot's
        privatized value; else sample uniformly from D\{pivot_value}.
    """
    cols = list(df_B.columns)
    d = len(cols)
    out = []

    for _, row in df_B.iterrows():
        j = np.random.randint(d)
        pivot_col = cols[j]
        y = {}

        # Pivot: GRR(ε)
        dom_pivot = domains[pivot_col]
        y_pivot = grr_perturb(row[pivot_col], dom_pivot, epsilon)
        y[pivot_col] = y_pivot

        # Non-pivot attributes
        for i, col in enumerate(cols):
            if i == j:
                continue
            dom = domains[col]
            py = p_y_table.get((pivot_col, col), 0.5)
            if np.random.rand() < py:
                y[col] = y_pivot
            else:
                others = [v for v in dom if v != y_pivot]
                y[col] = np.random.choice(others)
        out.append(y)

    return pd.DataFrame(out, index=df_B.index, columns=cols)


# -------------------- Phase II: mixture-aware estimator --------------------

def corr_rr_estimate(perturbed_df, domains, epsilon, p_y_table, clamp=True, renorm=True):
    """
    Debias Phase-II reports (which are a mixture of: pivot GRR(ε) + copy/uniform from pivot)
    to recover per-column marginals.

    For each category v, solve A(v) * r(v) = b(v), then invert GRR:
      f_j(v) = (r_j(v) - q) / Δ

    where:
      r_j(v) = Pr[Y_j = v when j is pivot] = q + Δ f_j(v)
      p = e^ε/(e^ε + k - 1), q = 1/(e^ε + k - 1), Δ = p - q
      d = #attributes, k = |domain|
      A_{kk} = 1
      A_{k,a} = (k * p_{a->k} - 1) / (k - 1),   a != k
      b_k = d * \hat r_k^{obs}(v) - sum_{a != k} (1 - p_{a->k})/(k - 1)
    """
    cols = list(perturbed_df.columns)
    d_attrs = len(cols)

    # Common domain assumption:
    domain0 = domains[cols[0]]
    k_dom = len(domain0)
    for c in cols:
        if domains[c] != domain0:
            raise ValueError("Corr-RR assumes a common categorical domain across columns.")

    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + k_dom - 1)
    q = 1.0 / (exp_eps + k_dom - 1)
    Delta = p - q

    n2 = len(perturbed_df)
    if n2 == 0 or np.isclose(Delta, 0.0):
        return {c: {v: 1.0 / k_dom for v in domain0} for c in cols}

    # Empirical column marginals in Phase II
    r_obs = {c: perturbed_df[c].value_counts(normalize=True).to_dict() for c in cols}
    for c in cols:
        for v in domain0:
            r_obs[c].setdefault(v, 0.0)

    est_phase2 = {c: {v: 0.0 for v in domain0} for c in cols}

    # Solve per category v
    for v in domain0:
        A = np.zeros((d_attrs, d_attrs), dtype=float)
        b = np.zeros(d_attrs, dtype=float)

        for row_idx, kcol in enumerate(cols):
            # diagonal
            A[row_idx, row_idx] = 1.0
            rhs_adjust = 0.0
            for col_idx, acol in enumerate(cols):
                if acol == kcol:
                    continue
                py = p_y_table.get((acol, kcol), 0.5)
                A[row_idx, col_idx] = (k_dom * py - 1.0) / (k_dom - 1.0)
                rhs_adjust += (1.0 - py) / (k_dom - 1.0)
            b[row_idx] = d_attrs * r_obs[kcol][v] - rhs_adjust

        # Solve A r = b (fallback to least squares if singular)
        try:
            r = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            r, *_ = np.linalg.lstsq(A, b, rcond=None)

        # invert GRR and sanitize
        for idx, col in enumerate(cols):
            f_val = (r[idx] - q) / Delta
            if clamp:
                f_val = max(0.0, min(1.0, f_val))
            est_phase2[col][v] = float(f_val)

    if renorm:
        for col in cols:
            s = sum(est_phase2[col].values())
            if s > 0:
                for v in domain0:
                    est_phase2[col][v] /= s
            else:
                for v in domain0:
                    est_phase2[col][v] = 1.0 / k_dom

    return est_phase2


# -------------------- Combine Phase I & II --------------------

def combine_phase_estimates(est_I, est_II, n1, n2):
    """
    Weighted combination by counts.
    """
    combined = {}
    cols = list(est_I.keys())
    for col in cols:
        combined[col] = {}
        for v in est_I[col]:
            combined[col][v] = (n1 * est_I[col][v] + n2 * est_II[col][v]) / (n1 + n2)
    return combined
