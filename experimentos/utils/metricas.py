"""
metricas.py

Métricas para evaluación de Archetypal Analysis.
"""

import numpy as np

def reconstruction_error(X, X_hat):
    """
    Error global de reconstrucción.
    """
    return np.linalg.norm(X - X_hat, ord="fro")

def reconstruction_error_per_sample(X, X_hat):
    """
    Error por estrella (fila).
    """
    return np.linalg.norm(X - X_hat, axis=1)

def relative_error(X, X_hat):
    """
    Error relativo promedio.
    """
    num = np.linalg.norm(X - X_hat)
    den = np.linalg.norm(X) + 1e-8
    return num / den

def compression_ratio(K, n_samples):
    """
    Proporción de compresión del modelo.
    """
    return (K * n_samples) / (n_samples ** 2)

def evaluate_model(X, alpha, Z):

    X_hat = alpha @ Z

    return {
        "error_fro": reconstruction_error(X, X_hat),
        "error_rel": relative_error(X, X_hat),
        "error_per_sample": reconstruction_error_per_sample(X, X_hat),
    }

