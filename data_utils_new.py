# data_utils_new.py
import numpy as np
import pandas as pd

def _as_vec(domain, spec):
    if isinstance(spec, dict):
        v = np.array([float(spec.get(d, 0.0)) for d in domain], dtype=float)
    else:
        v = np.array(spec, dtype=float)
        if len(v) != len(domain):
            raise ValueError("Probability list length must match domain size.")
    if (v < 0).any():
        raise ValueError("Probabilities must be non-negative.")
    s = v.sum()
    if s <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    return v / s

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
    Generate correlated pairs on a single numeric domain.
      - For each provided odd marginal p_i:
          X_{k}   ~ p_i, where k = start_index + 2*i
          X_{k+1} = X_{k} w.p. rho; else ~ q_marginal (defaults to uniform)
    If `odd_marginals` is a dict/list for a SINGLE marginal, produces just (X1, X2).

    Returns: DataFrame with columns X{start_index}..X{start_index+2*m-1}
    """
    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    d = len(domain)

    # Normalize inputs
    if isinstance(odd_marginals, (dict, list, tuple)) and not (
        isinstance(odd_marginals, list) and odd_marginals and isinstance(odd_marginals[0], (dict, list, tuple))
    ):
        # Single marginal -> one pair
        odd_list = [odd_marginals]
    else:
        # Already a list of marginals
        odd_list = list(odd_marginals)

    q = _as_vec(domain, q_marginal if q_marginal is not None else np.ones(d) / d)
    rho = float(np.clip(rho, 0.0, 1.0))

    data = {}
    for i, p_spec in enumerate(odd_list):
        odd_idx = start_index + 2 * i
        even_idx = odd_idx + 1
        odd_col, even_col = f"X{odd_idx}", f"X{even_idx}"

        p = _as_vec(domain, p_spec)

        # X_odd ~ p
        X_odd = np.array(domain)[np.random.choice(d, size=n, p=p)]

        # X_even: copy-or-q
        copy_mask = (np.random.rand(n) < rho)
        X_even = np.empty(n, dtype=object)
        X_even[copy_mask] = X_odd[copy_mask]
        non_idx = np.where(~copy_mask)[0]
        if non_idx.size:
            X_even[non_idx] = np.array(domain)[np.random.choice(d, size=non_idx.size, p=q)]

        data[odd_col]  = X_odd.astype(int)
        data[even_col] = X_even.astype(int)

    return pd.DataFrame(data)

# --------- small helpers ----------
def match_rate(df, a="X1", b="X2"):
    return float((df[a] == df[b]).mean())

def empirical_corr(df, a="X1", b="X2"):
    return df[a].astype(float).corr(df[b].astype(float))

def freqs(df, domain):
    out = {}
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, sort=False)
        out[col] = {v: float(vc.get(v, 0.0)) for v in domain}
    return out

def get_true_frequencies(df, columns=None):
    columns = columns or list(df.columns)
    out = {}
    for col in columns:
        counts = df[col].value_counts(normalize=True).sort_index()
        out[col] = counts.to_dict()
    return out




# # data_utils_new.py
# import numpy as np
# import pandas as pd

# # -------------------- core generator --------------------
# def gen_two_drifting(n, domain, x1_marginal, rho, q_marginal=None, seed=None):
#     """
#     Generate two correlated discrete vars X1, X2 on the SAME numeric domain.
#       - X1 ~ p (your x1_marginal)
#       - With prob rho:  X2 = X1
#         Else:           X2 ~ q  (defaults to uniform over domain)
#     => Marginal(X2) = rho * p + (1 - rho) * q  (X2 drifts toward q as rho↓).

#     Args:
#       n: number of samples
#       domain: list like [0,1,2,...] (must be numeric)
#       x1_marginal: dict or list aligned with domain (p)
#       rho: desired copy probability in [0,1]
#       q_marginal: dict or list aligned with domain for q (defaults to uniform)
#       seed: RNG seed

#     Returns: DataFrame with int columns X1, X2
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     domain = list(domain)
#     d = len(domain)

#     def as_vec(spec):
#         if isinstance(spec, dict):
#             v = np.array([float(spec.get(v, 0.0)) for v in domain], dtype=float)
#         else:
#             v = np.array(spec, dtype=float)
#             if len(v) != d:
#                 raise ValueError("Probability list length must match domain size.")
#         if (v < 0).any(): raise ValueError("Probabilities must be non-negative.")
#         s = v.sum()
#         if s <= 0: raise ValueError("Probabilities must sum to a positive value.")
#         return v / s

#     p = as_vec(x1_marginal)
#     q = as_vec(q_marginal if q_marginal is not None else np.ones(d) / d)
#     rho = float(np.clip(rho, 0.0, 1.0))

#     # Sample X1 ~ p
#     X1 = np.array(domain)[np.random.choice(d, size=n, p=p)]

#     # Copy-or-q for X2
#     copy_mask = (np.random.rand(n) < rho)
#     X2 = np.empty(n, dtype=object)
#     X2[copy_mask] = X1[copy_mask]
#     non_idx = np.where(~copy_mask)[0]
#     if non_idx.size:
#         X2[non_idx] = np.array(domain)[np.random.choice(d, size=non_idx.size, p=q)]

#     # Ensure numeric dtype for compatibility elsewhere
#     return pd.DataFrame({"X1": X1.astype(int), "X2": X2.astype(int)})

# # -------------------- utilities --------------------
# def match_rate(df):
#     """Fraction of rows where X1 == X2."""
#     return float((df["X1"] == df["X2"]).mean())

# def empirical_corr(df):
#     """Pearson correlation between numeric X1 and X2."""
#     return df["X1"].astype(float).corr(df["X2"].astype(float))

# def freqs(df, domain):
#     """Empirical marginals by column in the provided domain order (no sorting)."""
#     out = {}
#     for col in df.columns:
#         vc = df[col].value_counts(normalize=True, sort=False)
#         out[col] = {v: float(vc.get(v, 0.0)) for v in domain}
#     return out

# def get_true_frequencies(df, columns=None):
#     """
#     Normalized frequency dict per column (sorted by value to keep numeric order stable):
#       {col: {value: prob, ...}, ...}
#     """
#     columns = columns or list(df.columns)
#     out = {}
#     for col in columns:
#         counts = df[col].value_counts(normalize=True).sort_index()
#         out[col] = counts.to_dict()
#     return out
