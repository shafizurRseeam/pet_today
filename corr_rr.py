import numpy as np
import pandas as pd
from grr import grr_perturb, grr_estimate_frequencies
from data_utils import correlated_data_generator, get_true_frequencies  # NEW

# --- p_y Optimization (unchanged) ---
def optimal_p_y(f_a, f_b, epsilon, n, domain):
    d = len(domain)
    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + d - 1)
    q = 1.0 / (exp_eps + d - 1)
    Δ = p - q

    S1 = d*(d-1)/2
    S2 = (d-1)*d*(2*d-1)/6

    μa  = sum(v * f_a[v] for v in domain)
    μb  = sum(v * f_b[v] for v in domain)
    νb2 = sum(v**2 * f_b[v] for v in domain)

    a0 = μa - μb
    a1 = 2*μb - S1
    b1 = 2*νb2 - S2
    Y0 = (Δ/2)*a0 + (S1/2)

    α1 = 2 * sum((1 - f_a[v] - f_b[v]) * ((2*f_a[v]-1) + (2*f_b[v]-1)) for v in domain)
    α2 = sum((2*f_a[v]-1)**2 + (2*f_b[v]-1)**2 for v in domain)

    num = (b1 - 2*Y0*a1)/(2*n*Δ) + α1/(8*d)
    den = a1**2/(2*n) - α2/(4*d)

    p_star = num/den if den != 0 else 1
    return float(np.clip(p_star, 0.0, 1.0))

# --- Phase 1 SPL (unchanged) ---
def corr_rr_phase1_spl(df, epsilon, frac=0.1):
    n = len(df)
    m = int(frac * n)
    df_A = df.iloc[:m]
    eps_split = epsilon / df.shape[1]

    reports = []
    for _, row in df_A.iterrows():
        reports.append([
            grr_perturb(row[col], df[col].unique(), eps_split)
            for col in df.columns
        ])
    reports = np.array(reports, dtype=object)

    est = {}
    for i, col in enumerate(df.columns):
        domain = df[col].unique()
        est[col] = grr_estimate_frequencies(reports[:, i], domain, eps_split)
    return est, df.iloc[m:]

# --- Phase 2 + Estimation (unchanged) ---
def corr_rr_phase2_perturb(df, epsilon, f_hat_phase1, domain_map, p_y_table):
    d = len(df.columns)
    privatized = []
    for _, row in df.iterrows():
        j = np.random.randint(d)
        perturbed = {}
        for i, col in enumerate(df.columns):
            domain = domain_map[col]
            if i == j:
                perturbed[col] = grr_perturb(row[col], domain, epsilon)
            else:
                pair = (df.columns[j], col)
                py = p_y_table.get(pair, 0.5)
                if np.random.rand() < py:
                    perturbed[col] = row[col]
                else:
                    perturbed[col] = np.random.choice([v for v in domain if v != row[col]])
        privatized.append(perturbed)
    return pd.DataFrame(privatized, index=df.index)

def corr_rr_estimate(perturbed_df, domains, epsilon):
    estimates = {}
    n, d = perturbed_df.shape
    for col, domain in domains.items():
        reports = perturbed_df[col].tolist()
        estimates[col] = grr_estimate_frequencies(reports, domain, epsilon)
    return estimates

def combine_phase_estimates(est_A, est_B, m, n_minus_m):
    combined = {}
    for col in est_A:
        combined[col] = {}
        for v in est_A[col]:
            combined[col][v] = (m * est_A[col][v] + n_minus_m * est_B[col][v]) / (m + n_minus_m)
    return combined
