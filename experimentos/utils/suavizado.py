"""
suavizado.py

Funciones de suavizado opcional para los experimentos.
"""

import numpy as np

from config import SAVGOL_POLY, SAVGOL_WINDOW, SUAVIZADO


def sin_suavizado(X):
    return X


def savgol(X, window=5, poly=2):
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise ImportError(
            "Para usar SUAVIZADO='savgol' instala scipy: pip install scipy"
        ) from exc

    X = np.asarray(X)
    n_variables = X.shape[1]
    window = int(window)
    poly = int(poly)

    if window % 2 == 0:
        window += 1
    if window > n_variables:
        window = n_variables if n_variables % 2 == 1 else n_variables - 1
    if window <= poly:
        window = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    if window > n_variables:
        raise ValueError(
            f"Window Savitzky-Golay invalida: window={window}, n_variables={n_variables}"
        )

    return savgol_filter(X, window_length=window, polyorder=poly, axis=1)


def suavizar(X):
    if SUAVIZADO == "none":
        return sin_suavizado(X)

    if SUAVIZADO == "savgol":
        return savgol(X, window=SAVGOL_WINDOW, poly=SAVGOL_POLY)

    raise ValueError(f"Suavizado no soportado: {SUAVIZADO}")
