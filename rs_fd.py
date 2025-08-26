import numpy as np
import pandas as pd
from grr import grr_perturb

def rs_fd_perturb(df, domains, epsilon, seed=None):
    n, d = df.shape
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    cols = list(domains.keys())
    privatized = []

    for _, row in df.iterrows():
        j = np.random.randint(d)  # sampled attribute index
        out = {}
        for i, col in enumerate(cols):
            dom = domains[col]
            if i == j:
                out[col] = grr_perturb(row[col], dom, epsilon)      # GRR(ε)
            else:
                out[col] = np.random.choice(dom)                    # uniform fake
        privatized.append(out)

    if seed is not None:
        np.random.set_state(rng_state)

    return pd.DataFrame(privatized, index=df.index)

def rs_fd_estimate(perturbed_df, domains, epsilon):
    n, d = perturbed_df.shape
    estimates = {}

    for col, dom in domains.items():
        kj = len(dom)
        exp_eps = np.exp(epsilon)
        p = exp_eps / (exp_eps + kj - 1)
        q = 1.0 / (exp_eps + kj - 1)

        counts = perturbed_df[col].value_counts().to_dict()

        est = {}
        for v in dom:
            Ni = counts.get(v, 0)
            # debias with explicit mixture correction
            est[v] = (Ni * d * kj - n * (d - 1 + q * kj)) / (n * kj * (p - q))
        estimates[col] = est

    return estimates
