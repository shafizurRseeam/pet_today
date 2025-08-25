def compute_mse(true_counts, estimated_counts):
    """
    Mean Squared Error between two categorical distributions over same domain.
    """
    if true_counts.keys() != estimated_counts.keys():
        raise ValueError("Domains of true and estimated counts do not match.")
    return sum((true_counts[v] - estimated_counts[v])**2 for v in true_counts) / len(true_counts)
