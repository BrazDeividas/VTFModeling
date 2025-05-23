import numpy as np

def compute_interpolation_adjacency_matrix(distance_matrix, sigma = 0.1, epsilon = 0.5):
    d = distance_matrix / 100.
    d2 = d * d
    n = distance_matrix.shape[0]
    w_mask = np.ones([n, n]) - np.identity(n)
    return np.exp(-d2 / sigma ** 2) * (np.exp(-d2 / sigma ** 2) >= epsilon) * w_mask