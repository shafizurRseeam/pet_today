import numpy as np
import pandas as pd

def _as_prob_vector(domain, spec):
    """
    Convert a prob spec (dict {val:p} or list aligned with domain) to a vector
    aligned with `domain`, and validate it.
    """
    d = len(domain)
    if spec is None:
        v = np.ones(d, dtype=float) / d
    elif isinstance(spec, dict):
        v = np.array([float(spec.get(v, 0.0)) for v in domain], dtype=float)
    else:
        v = np.array(spec, dtype=float)
        if len(v) != d:
            raise ValueError("Probability list length must match domain size.")
    if (v < 0).any():
        raise ValueError("Probabilities must be non-negative.")
    s = v.sum()
    if s <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    return v / s

def _choice_from_probs(domain, p, size):
    idx = np.random.choice(len(domain), size=size, p=p)
    return np.array(domain, dtype=object)[idx]

def correlated_data_generator(
    domain,
    n,
    correlations=None,
    total_attributes=None,
    seed=None,
    base_marginals=None,
    default_base=None,
):
    """
    Generate a discrete dataset with optional pairwise 'same-value' correlations
    and NON-UNIFORM attribute marginals.

    Args:
      domain: iterable of categorical values (e.g., [0,1,2]).
      n: number of rows.
      correlations: list of tuples [(attr1, attr2, p_same), ...].
      total_attributes: ensure columns X1..X{total_attributes} exist.
      seed: RNG seed for reproducibility.
      base_marginals:
         - dict mapping attribute -> prob spec (dict {val:p} OR list aligned to domain)
         - optional key 'default' for fallback.
      default_base:
         - global default prob spec if not provided in base_marginals['default'].
           If both missing, uniform is used.
    """
    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    d = len(domain)

    # Build per-attribute base distributions
    def get_base(attr):
        if isinstance(base_marginals, dict) and attr in base_marginals:
            return _as_prob_vector(domain, base_marginals[attr])
        if isinstance(base_marginals, dict) and 'default' in base_marginals:
            return _as_prob_vector(domain, base_marginals['default'])
        if default_base is not None:
            return _as_prob_vector(domain, default_base)
        return np.ones(d) / d  # uniform

    df = pd.DataFrame()

    # Process correlation pairs; reuse existing attr1 if already present
    if correlations:
        for attr1, attr2, p_same in correlations:
            # Ensure X_attr1 exists (sample from its base marginal)
            if attr1 not in df.columns:
                p1 = get_base(attr1)
                df[attr1] = _choice_from_probs(domain, p1, n)
            X1 = df[attr1].to_numpy()

            # Prepare base for attr2
            p2_full = get_base(attr2)

            # Decide where to copy vs differ
            mask_same = np.random.rand(n) < float(p_same)
            X2 = np.empty(n, dtype=object)

            # Copy positions
            X2[mask_same] = X1[mask_same]

            # Different positions: sample from p2 restricted to values != X1[i]
            if (~mask_same).any():
                idxs = np.where(~mask_same)[0]
                x1_diff = X1[idxs]
                # For each row, zero out the prob of X1[i] then renormalize
                val_to_idx = {v: i for i, v in enumerate(domain)}
                for j, v1 in enumerate(x1_diff):
                    p = p2_full.copy()
                    p[val_to_idx[v1]] = 0.0
                    s = p.sum()
                    if s == 0:
                        # degenerate: if base put all mass on v1, fallback to uniform over others
                        p[:] = 1.0
                        p[val_to_idx[v1]] = 0.0
                        p /= p.sum()
                    else:
                        p /= s
                    X2[idxs[j]] = _choice_from_probs(domain, p, 1)[0]

            df[attr2] = X2

    # Ensure requested attributes exist (with their own marginals)
    if total_attributes is not None:
        all_attrs = [f'X{i+1}' for i in range(total_attributes)]
    else:
        all_attrs = list(df.columns)

    for attr in all_attrs:
        if attr not in df.columns:
            p = get_base(attr)
            df[attr] = _choice_from_probs(domain, p, n)

    # Reorder columns if total_attributes was provided
    if total_attributes is not None:
        df = df[[f'X{i+1}' for i in range(total_attributes)]]

    return df

def get_true_frequencies(df, columns=None):
    columns = columns or list(df.columns)
    out = {}
    for col in columns:
        counts = df[col].value_counts(normalize=True).sort_index()
        out[col] = counts.to_dict()
    return out
