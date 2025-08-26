import numpy as np
import pandas as pd
from grr import grr_perturb

def _normalized_prior(domain, prior_dict):
    probs = np.array([max(0.0, float(prior_dict.get(v, 0.0))) for v in domain], dtype=float)
    s = probs.sum()
    if s <= 0:
        # fallback to uniform if prior is degenerate/missing
        return np.full(len(domain), 1.0 / len(domain))
    return probs / s

def rs_rfd_perturb(df, domains, priors, epsilon, seed=None):
    """
    RS+RFD client-side perturbation:
      - sample one attribute uniformly
      - perturb sampled attr with GRR(ε)
      - set others by sampling from provided priors
    Returns a DataFrame aligned with df.index/columns.
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    cols = list(df.columns)
    d = len(cols)
    out_rows = []

    for _, row in df.iterrows():
        j = np.random.randint(d)  # sampled attribute index
        y = {}
        for i, col in enumerate(cols):
            domain = list(domains[col])
            if i == j:
                y[col] = grr_perturb(row[col], domain, epsilon)
            else:
                p_prior = _normalized_prior(domain, priors.get(col, {}))
                y[col] = np.random.choice(domain, p=p_prior)
        out_rows.append(y)

    if seed is not None:
        np.random.set_state(rng_state)

    return pd.DataFrame(out_rows, index=df.index, columns=cols)

def rs_rfd_estimate(perturbed_df, domains, priors, epsilon):
    """
    RS+RFD server-side estimator:
      r = (1/d)[q + (p-q) f] + ((d-1)/d) * prior
      => f = [d*r - q - (d-1)*prior] / (p - q)
    """
    n, d = perturbed_df.shape
    estimates = {}

    # handle ε = 0 (no signal): fall back to the prior
    if np.isclose(epsilon, 0.0):
        for col, domain in domains.items():
            p_prior = _normalized_prior(domain, priors.get(col, {}))
            estimates[col] = {v: float(p_prior[i]) for i, v in enumerate(domain)}
        return estimates

    for col, domain in domains.items():
        domain = list(domain)
        kj = len(domain)
        exp_eps = np.exp(epsilon)
        p = exp_eps / (exp_eps + kj - 1)
        # Either expression for q is fine; use the algebraically stable one:
        q = 1.0 / (exp_eps + kj - 1)

        counts = perturbed_df[col].value_counts().to_dict()
        prior_vec = _normalized_prior(domain, priors.get(col, {}))

        est_col = {}
        denom = (p - q)
        for i, v in enumerate(domain):
            Cvi = counts.get(v, 0)
            r = Cvi / n
            bias = q + (d - 1) * prior_vec[i]
            est_col[v] = (d * r - bias) / denom
        # optional: clamp to [0,1] and renormalize
        # clamp
        est_col = {v: max(0.0, min(1.0, est_col[v])) for v in domain}
        s = sum(est_col.values())
        if s > 0:
            est_col = {v: est_col[v] / s for v in domain}
        estimates[col] = est_col

    return estimates
