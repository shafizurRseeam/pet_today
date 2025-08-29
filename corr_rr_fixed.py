# corr_rr_fixed.py
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
# p_y optimizer (GENERAL k, exact, includes (p+q-1) term)
# ---------------------------------------------------------------------

def optimal_p_y(f_a, f_b, epsilon, n2, domain):
    """
    Exact closed-form minimizer of the average Phase-II MSE (both attributes),
    for general k-ary GRR. This includes the (p+q-1) term that vanishes only
    in the binary case (k=2).

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

    # GRR params
    exp_eps = np.exp(float(epsilon))
    p = exp_eps / (exp_eps + k - 1.0)
    q = 1.0     / (exp_eps + k - 1.0)
    Delta = p - q

    # Sums for the closed form
    sum_e2   = 0.0
    sum_term = 0.0

    for v in domain:
        fa = float(f_a.get(v, 0.0))
        fb = float(f_b.get(v, 0.0))
        e  = 2.0*fb - 1.0
        d0 = 1.0 - fa - fb
        a0 = fa - fb

        # numerator aggregate: e * [ (p+q-1)/(n2*Delta) + a0/n2 - d0 ]
        sum_term += e * ( (p + q - 1.0)/(n2*Delta) + (a0/n2) - d0 )
        sum_e2   += e * e

    denom = (1.0 - 1.0/n2) * sum_e2
    if abs(denom) < 1e-12:
        # No signal in e(v): default to neutral copy-prob
        return 0.5

    p_star = sum_term / denom
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

    for j in cols:
        for k in cols:
            if j == k:
                continue
            domain = domain_map[j]  # same size for all
            f_a = _safe_get_freqs(f_hat_phase1[j], domain)
            f_b = _safe_get_freqs(f_hat_phase1[k], domain)
            py  = optimal_p_y(f_a, f_b, epsilon=epsilon, n2=n2, domain=domain)
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
    p_y_table = build_p_y_table(f_hat_I, n2=n2, domain_map=domain_map, epsilon=epsilon)
    df_II = corr_rr_phase2_perturb(df_B, epsilon, f_hat_I, domain_map, p_y_table, rng=rng)
    est_II = corr_rr_estimate(df_II, domain_map, epsilon)
    combined = combine_phase_estimates(f_hat_I, est_II, n1=n2 and n1, n2=n2)  # keep exact weighting
    return combined, p_y_table, (n1, n2)

# ---------------------------------------------------------------------
# Extras: sweep & p_y table generation (optional)
# ---------------------------------------------------------------------

def _normalize_dist(d):
    vals = np.array([max(0.0, float(v)) for v in d.values()], dtype=float)
    s = vals.sum()
    if s <= 0:
        k = len(vals)
        vals = np.full(k, 1.0 / k)
    else:
        vals = vals / s
    return {k: vals[i] for i, k in enumerate(d.keys())}

def _build_p_y_table_minimal(est_I, epsilon, n2, domain, cols):
    """
    Build p_y[(a->b)] for all ordered pairs using optimal_p_y().
    """
    return {
        (a, b): float(optimal_p_y(est_I[a], est_I[b], epsilon, n2, domain))
        for a in cols for b in cols if a != b
    }

# ---------------- p_y table viewer ----------------

def p_y_tables_for_epsilons(
    df,
    epsilons,
    frac_phase1_corr=0.1,
    use_minimal_builder=True,
    csv_dir=None,
    float_fmt="%.6f",
):
    """
    For each epsilon:
      - run Phase I via corr_rr_phase1_spl
      - build p_y[(a->b)] using either _build_p_y_table_minimal(...) or build_p_y_table(...)
      - pretty-print and (optionally) save a CSV matrix with rows a, cols b

    Returns: dict {epsilon: pandas.DataFrame} where DataFrame is the p_y matrix.
    """
    if csv_dir:
        import os
        os.makedirs(csv_dir, exist_ok=True)

    cols = list(df.columns)
    results = {}

    for eps in epsilons:
        # Phase I (SPL) to get est_I and stable domains
        est_I, df_B, doms_stable = corr_rr_phase1_spl(df, eps, frac=frac_phase1_corr)
        n2 = len(df_B)

        # choose builder
        if use_minimal_builder:
            common_domain = doms_stable[cols[0]]
            pmap = _build_p_y_table_minimal(est_I, eps, n2, common_domain, cols)
        else:
            pmap = build_p_y_table(est_I, n2, doms_stable, epsilon=eps)

        # pivot to a matrix with NaN on diagonal
        mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for a in cols:
            for b in cols:
                if a == b:
                    mat.loc[a, b] = float('nan')
                else:
                    mat.loc[a, b] = float(pmap[(a, b)])

        # pretty print
        print(f"\n=== p_y table (Corr-RR Phase I → optimization) for epsilon = {eps} ===")
        with pd.option_context('display.float_format', lambda v: float_fmt % v):
            print(mat)

        # optional save
        if csv_dir:
            import os
            fname = os.path.join(csv_dir, f"py_table_eps_{str(eps).replace('.', '_')}.csv")
            mat.to_csv(fname, float_format=float_fmt)
            print(f"[saved] {fname}")

        results[eps] = mat

    return results
