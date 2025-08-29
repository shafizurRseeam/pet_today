import numpy as np
import pandas as pd

def correlated_data_generator(domain, n, correlations=None, total_attributes=None, seed=None):
    """
    Generate a discrete dataset with optional pairwise 'same-value' correlations.
    - domain: iterable of categorical values (e.g., [0,1]).
    - correlations: list of tuples [(attr1, attr2, p_same), ...]
      Each pair forces X_attr2 == X_attr1 with prob p_same, else random different.
    - total_attributes: if provided, ensures columns X1..X{total_attributes} exist,
      filling any missing ones with i.i.d. draws from domain.
    """
    if seed is not None:
        np.random.seed(seed)

    domain = list(domain)
    domain_set = {x: [d for d in domain if d != x] for x in domain}
    df = pd.DataFrame()

    # correlated pairs
    if correlations:
        for attr1, attr2, p_same in correlations:
            X1 = np.random.choice(domain, size=n)
            mask = np.random.rand(n) < p_same
            X2 = np.where(mask, X1, [np.random.choice(domain_set[v]) for v in X1])
            df[attr1] = X1
            df[attr2] = X2

    # ensure requested attributes exist
    if total_attributes is not None:
        all_attrs = [f'X{i+1}' for i in range(total_attributes)]
    else:
        all_attrs = list(df.columns)

    for attr in all_attrs:
        if attr not in df.columns:
            df[attr] = np.random.choice(domain, size=n)

    return df


def get_true_frequencies(df, columns=None):
    """
    Return normalized frequency dict per column:
      {col: {value: prob, ...}, ...}
    """
    columns = columns or list(df.columns)
    out = {}
    for col in columns:
        counts = df[col].value_counts(normalize=True).sort_index()
        out[col] = counts.to_dict()
    return out
