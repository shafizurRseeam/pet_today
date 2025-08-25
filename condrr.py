import numpy as np
import pandas as pd

def _binary_grr_p(eps):
    return np.exp(eps) / (1.0 + np.exp(eps))

def _binary_grr_matrix(eps):
    p = _binary_grr_p(eps)
    return np.array([[p, 1.0 - p],
                     [1.0 - p, p]], dtype=float)

def _binary_grr_matrix_inv(eps):
    p = _binary_grr_p(eps)
    denom = (2.0 * p - 1.0)
    if np.isclose(denom, 0.0):
        denom = 1e-12
    return (1.0 / denom) * np.array([[p, -(1.0 - p)],
                                     [-(1.0 - p), p]], dtype=float)

def _normalize_rows(mat):
    out = mat.copy().astype(float)
    for r in range(out.shape[0]):
        s = out[r].sum()
        out[r] = 0.5 if s <= 0 else out[r] / s
    return out

def _normalize_cols(mat):
    out = mat.copy().astype(float)
    for c in range(out.shape[1]):
        s = out[:, c].sum()
        out[:, c] = 0.5 if s <= 0 else out[:, c] / s
    return out

def _clip_renorm_joint(j):
    x = np.clip(j, 0.0, None)
    s = x.sum()
    return np.full((2, 2), 0.25) if s <= 0 else x / s

def _sample_from_probs(probs):
    p = np.clip(np.array(probs, dtype=float), 0.0, None)
    s = p.sum()
    if s <= 0:
        p = np.array([0.5, 0.5])
    else:
        p = p / s
    return np.random.choice([0, 1], p=p)

def condrr_perturb_binary(df, epsilon, n1=None, frac=0.2):
    """
    Phase I: first n1 (or frac) users with ε/2 per bit -> debias joint -> build theta tables.
    Phase II: remaining users pivot one bit with GRR(ε) and synthesize the other via theta.
    Returns (reports, public_info).
    """
    assert df.shape[1] == 2, "Cond-RR (this impl) expects exactly 2 binary attributes."
    col1, col2 = df.columns[0], df.columns[1]
    dom1 = sorted(df[col1].unique().tolist())
    dom2 = sorted(df[col2].unique().tolist())
    v2i1 = {v: i for i, v in enumerate(dom1)}
    v2i2 = {v: i for i, v in enumerate(dom2)}
    i2v1 = {i: v for i, v in enumerate(dom1)}
    i2v2 = {i: v for i, v in enumerate(dom2)}

    n = len(df)
    if n1 is None:
        n1 = max(1, int(np.floor(frac * n)))
    n1 = min(n1, n)
    phase1_idx = np.arange(n)[:n1]
    phase2_idx = np.arange(n)[n1:]

    eps_I = epsilon / 2.0
    M_I = _binary_grr_matrix(eps_I)
    M_I_inv = _binary_grr_matrix_inv(eps_I)

    counts_Y = np.zeros((2, 2), dtype=float)
    pI = _binary_grr_p(eps_I)
    for idx in phase1_idx:
        x1 = v2i1[df.iloc[idx, 0]]
        x2 = v2i2[df.iloc[idx, 1]]
        y1 = x1 if (np.random.rand() < pI) else (1 - x1)
        y2 = x2 if (np.random.rand() < pI) else (1 - x2)
        counts_Y[y1, y2] += 1.0

    Y_hat = counts_Y / max(1.0, len(phase1_idx))
    X_vec_hat = (np.kron(M_I_inv, M_I_inv) @ Y_hat.reshape(-1, order='C'))
    X_hat = _clip_renorm_joint(X_vec_hat.reshape(2, 2, order='C'))

    M_II = _binary_grr_matrix(epsilon)
    tilde = (np.kron(M_II, M_II) @ X_hat.reshape(-1, order='C')).reshape(2, 2, order='C')

    theta_2_given_1 = _normalize_rows(tilde)  # rows: y1, cols: y2
    theta_1_given_2 = _normalize_cols(tilde)  # rows: y1, cols: y2; but used column-wise

    pII = _binary_grr_p(epsilon)
    reports = []
    for idx in phase2_idx:
        x1 = v2i1[df.iloc[idx, 0]]
        x2 = v2i2[df.iloc[idx, 1]]
        pivot = np.random.choice([0, 1])  # uniform
        if pivot == 0:
            y1 = x1 if (np.random.rand() < pII) else (1 - x1)
            y2 = _sample_from_probs(theta_2_given_1[y1, :])
        else:
            y2 = x2 if (np.random.rand() < pII) else (1 - x2)
            # theta_1_given_2 is column-normalized of tilde, so P(Y1|Y2)
            y1 = _sample_from_probs(theta_1_given_2[:, y2])
        reports.append([i2v1[y1], i2v2[y2]])

    public = {
        'phase1_idx': phase1_idx,
        'phase2_idx': phase2_idx,
        'theta_2_given_1': theta_2_given_1,
        'theta_1_given_2': theta_1_given_2,
        'maps': {'dom1': dom1, 'dom2': dom2, 'v2i1': v2i1, 'v2i2': v2i2, 'i2v1': i2v1, 'i2v2': i2v2},
    }
    return reports, public

def condrr_estimate_binary(reports, df, epsilon):
    """
    Debias each attribute marginal as GRR(ε).
    """
    arr = np.asarray(reports, dtype=object)
    col1, col2 = df.columns[0], df.columns[1]
    dom1 = sorted(df[col1].unique().tolist())
    dom2 = sorted(df[col2].unique().tolist())

    # standard GRR debias for binary
    exp_eps = np.exp(epsilon)
    p = exp_eps / (1 + exp_eps)
    q = 1 - p
    denom = (p - q)

    def _est(col_ix, domain):
        counts = {v: 0 for v in domain}
        for r in arr[:, col_ix]:
            counts[r] += 1
        n = len(arr)
        return {v: (counts[v] - n * q) / (n * denom) for v in domain}

    return {col1: _est(0, dom1), col2: _est(1, dom2)}
