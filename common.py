import numpy as np


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    return (X - mean) / std, mean, std


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
