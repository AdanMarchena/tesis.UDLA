"""
inicializacion.py
"""

import numpy as np
from sklearn.decomposition import PCA


# Inicializacion de Beta random
def init_random_beta(n_samples, K, seed=42):
    rng = np.random.default_rng(seed)
    beta = rng.random((K, n_samples))
    beta /= beta.sum(axis=1, keepdims=True)
    return beta


# Inicializacion de Beta KMeans
def init_kmeans_beta(X, K):
    raise NotImplementedError("Pendiente implementar kmeans initialization")


# Inicializacion de Beta furthest-sum
def init_furthest_sum_beta(X, K, seed=42, return_indices=False):
    X = np.asarray(X)
    n_samples = X.shape[0]

    if K > n_samples:
        raise ValueError(f"K ({K}) no puede ser mayor que n_samples ({n_samples})")

    rng = np.random.default_rng(seed)
    seleccionados = [int(rng.integers(n_samples))]

    while len(seleccionados) < K:
        distancias = np.linalg.norm(X[:, None, :] - X[seleccionados][None, :, :], axis=2)
        suma_distancias = distancias.sum(axis=1)
        suma_distancias[seleccionados] = -np.inf
        seleccionados.append(int(np.argmax(suma_distancias)))

    beta = np.zeros((K, n_samples))
    beta[np.arange(K), seleccionados] = 1.0

    if return_indices:
        return beta, seleccionados

    return beta


# Inicializacion de Beta PCA
def init_pca_beta(X, K, seed=42):
    X = np.asarray(X)
    n_samples = X.shape[0]

    if K > n_samples:
        raise ValueError(f"K ({K}) no puede ser mayor que n_samples ({n_samples})")

    coords = PCA(n_components=2).fit_transform(X)
    candidatos = [
        int(np.argmin(coords[:, 0])),
        int(np.argmax(coords[:, 0])),
        int(np.argmin(coords[:, 1])),
        int(np.argmax(coords[:, 1])),
    ]

    seleccionados = []
    for idx in candidatos:
        if idx not in seleccionados:
            seleccionados.append(idx)
        if len(seleccionados) == K:
            break

    if len(seleccionados) < K:
        rng = np.random.default_rng(seed)
        restantes = np.setdiff1d(np.arange(n_samples), np.array(seleccionados))
        extra = rng.choice(restantes, size=K - len(seleccionados), replace=False)
        seleccionados.extend(int(idx) for idx in extra)

    beta = np.zeros((K, n_samples))
    beta[np.arange(K), seleccionados] = 1.0

    return beta


# Funcion principal para inicializar Beta segun la configuracion
def inicializar_beta(X, K, seed=42, metodo=None):
    if metodo is None:
        from config import INICIALIZACION

        metodo = INICIALIZACION

    if metodo == "random":
        return init_random_beta(X.shape[0], K, seed)

    elif metodo == "kmeans":
        return init_kmeans_beta(X, K)

    elif metodo == "furthest_sum":
        return init_furthest_sum_beta(X, K, seed)

    elif metodo == "pca":
        return init_pca_beta(X, K, seed)

    else:
        raise ValueError(f"Inicializacion no soportada: {metodo}")
