import numpy as np

def minmax(X):
    X_min = np.min(X, axis=1, keepdims=True)
    X_max = np.max(X, axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)

def centered(X):
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean

def zscore(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)