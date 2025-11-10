# data_utils_new.py
import numpy as np
import pandas as pd

# -------------------- internal utils --------------------
def _as_vec(domain, spec):
    """
    Convert a dict/list of probabilities to a normalized numpy vector aligned with `domain`.
    - dict: keys are domain values; missing keys treated as 0
    - list/array: must match len(domain)
    """
    domain = list(domain)
    k = len(domain)
    if isinstance(spec, dict):
        v = np.array([float(spec.get(d, 0.0)) for d in domain], dtype=float)
    else:
        v = np.array(spec, dtype=float)
        if len(v) != k:
            raise ValueError("Probability list length must match domain size.")
    if (v < 0).any():
        raise ValueError("Probabilities must be non-negative.")
    s = v.sum()
    if s <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    return v / s

def _maybe_int(arr):
    """Try to cast to int; if it fails (non-numeric domain), keep original dtype."""
    try:
        return arr.astype(int)
    except Exception:
        return arr

# =========================================================
#                   GENERATORS
# =========================================================

def gen_star_from_x1(
    n,
    domain,
    d,                  # total number of attributes (>=1)
    x1_marginal,        # marginal p for X1
    rho,                # global copy prob for ALL Xj (j>=2) relative to X1
    q_marginal=None,    # base q for non-copy draws; None -> uniform
    seed=None,
):
    """
    STAR model: X1 is the hub.
      - X1 ~ p  (p = x1_marginal)
      - For each j in {2..d}: Xj = X1 with prob rho; else Xj ~ q (independent)
    All attributes share the same domain.

    Returns: DataFrame with columns X1..Xd
    """
    if d < 1:
        raise ValueError("d must be at least 1.")

    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    k = len(domain)
    p = _as_vec(domain, x1_marginal)
    q = _as_vec(domain, q_marginal if q_marginal is not None else np.ones(k) / k)
    rho = float(np.clip(rho, 0.0, 1.0))

    # X1 ~ p
    X1 = np.array(domain)[np.random.choice(k, size=n, p=p)]
    data = {"X1": _maybe_int(X1)}

    # Xj for j >= 2: copy-or-q from X1 (independently per attribute)
    for j in range(2, d + 1):
        copy_mask = (np.random.rand(n) < rho)
        Xj = np.empty(n, dtype=object)
        Xj[copy_mask] = X1[copy_mask]
        non_idx = np.where(~copy_mask)[0]
        if non_idx.size:
            Xj[non_idx] = np.array(domain)[np.random.choice(k, size=non_idx.size, p=q)]
        data[f"X{j}"] = _maybe_int(Xj)

    return pd.DataFrame(data)

def gen_progressive(
    n,
    domain,
    d,                  # total number of attributes (>=1)
    x1_marginal,        # marginal p for X1
    rho,                # global copy prob for all Xj (j>=2)
    q_marginal=None,    # base q for non-copy draws; None -> uniform
    seed=None,
):
    """
    PROGRESSIVE model:
      - X1 ~ p  (p = x1_marginal)
      - For j >= 2:
            Choose a parent π(j) uniformly from {1,...,j-1}
            Xj = X_{π(j)} w.p. rho
            Xj = Z_j ~ q   w.p. 1 - rho

    Returns: DataFrame with columns X1..Xd
    """
    if d < 1:
        raise ValueError("d must be at least 1.")

    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    k = len(domain)

    # Normalize distributions
    p = _as_vec(domain, x1_marginal)
    q = _as_vec(domain, q_marginal if q_marginal is not None else np.ones(k) / k)
    rho = float(np.clip(rho, 0.0, 1.0))

    # --- 1. Sample X1 ~ p ---
    X = {}
    X1 = np.array(domain)[np.random.choice(k, size=n, p=p)]
    X["X1"] = _maybe_int(X1)

    # --- 2. Generate X2..Xd under progressive dependency ---
    for j in range(2, d + 1):
        # Parent index uniformly from {1, ..., j-1}
        parent = np.random.randint(1, j)  # inclusive of 1, exclusive of j

        copy_mask = (np.random.rand(n) < rho)
        Xj = np.empty(n, dtype=object)

        # Copy from parent
        parent_col = f"X{parent}"
        parent_vals = X[parent_col]
        Xj[copy_mask] = parent_vals[copy_mask]

        # Draw fresh from q
        non_idx = np.where(~copy_mask)[0]
        if non_idx.size:
            Xj[non_idx] = np.array(domain)[np.random.choice(k, size=non_idx.size, p=q)]

        X[f"X{j}"] = _maybe_int(Xj)

    return pd.DataFrame(X)

def gen_correlated_pairs(
    n,
    domain,
    rho,
    odd_marginals,          # dict for X1 only, OR list [p1, p3, p5, ...]
    q_marginal=None,        # base for even attrs when not copying; None -> uniform
    seed=None,
    start_index=1,
):
    """
    PAIR model: generate (X1,X2), (X3,X4), ... on a common domain.
      - For each provided odd marginal p_i:
          X_{k}   ~ p_i, where k = start_index + 2*i
          X_{k+1} = X_{k} w.p. rho; else ~ q
    If `odd_marginals` is a single dict/list -> only one pair (X1,X2).

    Returns: DataFrame with columns X{start_index}..X{start_index+2*m-1}
    """
    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    k = len(domain)

    # Normalize `odd_marginals` to a list of specs
    if isinstance(odd_marginals, (dict, list, tuple)) and not (
        isinstance(odd_marginals, list) and odd_marginals and isinstance(odd_marginals[0], (dict, list, tuple))
    ):
        odd_list = [odd_marginals]  # single pair
    else:
        odd_list = list(odd_marginals)

    q = _as_vec(domain, q_marginal if q_marginal is not None else np.ones(k) / k)
    rho = float(np.clip(rho, 0.0, 1.0))

    data = {}
    for i, p_spec in enumerate(odd_list):
        odd_idx = start_index + 2 * i
        even_idx = odd_idx + 1
        odd_col, even_col = f"X{odd_idx}", f"X{even_idx}"

        p = _as_vec(domain, p_spec)

        X_odd = np.array(domain)[np.random.choice(k, size=n, p=p)]
        copy_mask = (np.random.rand(n) < rho)
        X_even = np.empty(n, dtype=object)
        X_even[copy_mask] = X_odd[copy_mask]
        non_idx = np.where(~copy_mask)[0]
        if non_idx.size:
            X_even[non_idx] = np.array(domain)[np.random.choice(k, size=non_idx.size, p=q)]

        data[odd_col]  = _maybe_int(X_odd)
        data[even_col] = _maybe_int(X_even)

    return pd.DataFrame(data)


def gen_two_drifting(n, domain, x1_marginal, rho, q_marginal=None, seed=None):
    """
    TWO-VAR model: generate X1, X2 on the same domain.
      - X1 ~ p
      - X2 = X1 w.p. rho; else ~ q   (q uniform if None)
    Marginal(X2) = rho * p + (1 - rho) * q.

    Returns: DataFrame with columns X1, X2
    """
    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    k = len(domain)
    p = _as_vec(domain, x1_marginal)
    q = _as_vec(domain, q_marginal if q_marginal is not None else np.ones(k) / k)
    rho = float(np.clip(rho, 0.0, 1.0))

    X1 = np.array(domain)[np.random.choice(k, size=n, p=p)]
    copy_mask = (np.random.rand(n) < rho)
    X2 = np.empty(n, dtype=object)
    X2[copy_mask] = X1[copy_mask]
    non_idx = np.where(~copy_mask)[0]
    if non_idx.size:
        X2[non_idx] = np.array(domain)[np.random.choice(k, size=non_idx.size, p=q)]

    return pd.DataFrame({"X1": _maybe_int(X1), "X2": _maybe_int(X2)})

# =========================================================
#                   HELPERS
# =========================================================

def match_rate(df, a="X1", b="X2"):
    """Fraction of rows where column a equals column b."""
    return float((df[a] == df[b]).mean())

def match_rate_to_x1(df, j="X2"):
    """Fraction of rows where X1 equals Xj."""
    return float((df["X1"] == df[j]).mean())

def empirical_corr(df, a="X1", b="X2"):
    """Pearson correlation between columns a and b (cast to float)."""
    return df[a].astype(float).corr(df[b].astype(float))

def empirical_corr_to_x1(df, j="X2"):
    """Pearson correlation between X1 and Xj (cast to float)."""
    return df["X1"].astype(float).corr(df[j].astype(float))

def freqs(df, domain):
    """
    Empirical marginals for each column in the provided domain order
    (no sorting; missing values get probability 0).
    """
    out = {}
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, sort=False)
        out[col] = {v: float(vc.get(v, 0.0)) for v in domain}
    return out

def get_true_frequencies(df, columns=None):
    """
    Normalized frequency dict per column, sorted by the value (stable for numeric domains).
      {col: {value: prob, ...}, ...}
    """
    columns = columns or list(df.columns)
    out = {}
    for col in columns:
        counts = df[col].value_counts(normalize=True).sort_index()
        out[col] = counts.to_dict()
    return out
