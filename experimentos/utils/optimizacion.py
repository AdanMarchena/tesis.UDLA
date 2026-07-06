"""
optimizacion.py

Módulo de optimización para Archetypal Analysis.
"""

import numpy as np
from sklearn.decomposition import PCA
from config import (
    OPTIMIZACION,
    MAX_ITER,
    SEED,
    LR_ALPHA,
    LR_BETA,
    N_ALPHA_STEPS,
    N_BETA_STEPS,
    ALPHA_ITER,
    BETA_ITER,
    PATIENCE,
    ROBUST_LR_BETA,
)
from utils.inicializacion import inicializar_beta

def project_to_simplex(v):
    v = np.maximum(v, 0)
    s = v.sum()
    return v / (s + 1e-8)

def compute_archetypes(beta, X):
    return beta @ X

def project_rows_to_simplex(M):
    M_proj = np.zeros_like(M)

    for i in range(M.shape[0]):
        fila = project_to_simplex(M[i])
        if fila.sum() <= 1e-12:
            fila = np.full(M.shape[1], 1.0 / M.shape[1])
        M_proj[i] = fila

    return M_proj

def update_alpha(X, Z):

    n = X.shape[0]
    K = Z.shape[0]

    alpha = np.zeros((n, K))

    for i in range(n):

        a = np.linalg.lstsq(Z.T, X[i], rcond=None)[0]
        alpha[i] = project_to_simplex(a)

    return alpha

def update_beta(alpha, X):

    K = alpha.shape[1]
    n = X.shape[0]

    beta = np.zeros((K, n))

    for k in range(K):

        weights = alpha[:, k]
        b = weights.copy()

        beta[k] = project_to_simplex(b)

    return beta

def archetypal_analysis(X, K, max_iter=None, seed=42):

    if max_iter is None:
        max_iter = MAX_ITER

    beta = inicializar_beta(X, K, seed=seed)

    prev_error = np.inf

    for it in range(max_iter):

        Z = compute_archetypes(beta, X)
        alpha = update_alpha(X, Z)
        beta = update_beta(alpha, X)

        X_hat = alpha @ Z
        error = np.linalg.norm(X - X_hat, 'fro')

        print(f"Iter {it:03d} | error: {error:.6f}")

        # criterio de convergencia
        if abs(prev_error - error) < 1e-6:
            print("Convergencia alcanzada")
            break

        prev_error = error

    return alpha, beta, Z

def pgd_archetypal_analysis(
    X,
    K,
    max_iter=None,
    seed=42,
    lr_alpha=LR_ALPHA,
    lr_beta=LR_BETA,
    n_alpha_steps=N_ALPHA_STEPS,
    n_beta_steps=N_BETA_STEPS,
):

    if max_iter is None:
        max_iter = MAX_ITER

    n = X.shape[0]

    beta = inicializar_beta(X, K, seed=seed)
    alpha = np.full((n, K), 1.0 / K)

    prev_error = np.inf

    for it in range(max_iter):

        Z = compute_archetypes(beta, X)

        for _ in range(n_alpha_steps):
            grad_alpha = 2 * (alpha @ Z - X) @ Z.T
            alpha = project_rows_to_simplex(alpha - lr_alpha * grad_alpha)

        for _ in range(n_beta_steps):
            Z = compute_archetypes(beta, X)
            residual = alpha @ Z - X
            grad_beta = 2 * alpha.T @ residual @ X.T
            beta = project_rows_to_simplex(beta - lr_beta * grad_beta)

        Z = compute_archetypes(beta, X)
        X_hat = alpha @ Z
        error = np.linalg.norm(X - X_hat, 'fro')

        print(f"Iter {it:03d} | error: {error:.6f}")

        # criterio de convergencia
        if abs(prev_error - error) < 1e-6:
            print("Convergencia alcanzada")
            break

        prev_error = error

    return alpha, beta, Z

def _obtener_nnls():
    try:
        from scipy.optimize import nnls
    except ImportError as exc:
        raise ImportError(
            "Para usar OPTIMIZACION='nnls' instala scipy: pip install scipy"
        ) from exc

    return nnls

def nnls_alpha_beta_projected(X, K, max_iter=None, seed=42):

    if max_iter is None:
        max_iter = MAX_ITER

    nnls = _obtener_nnls()
    beta = inicializar_beta(X, K, seed=seed)

    prev_error = np.inf

    for it in range(max_iter):

        Z = compute_archetypes(beta, X)

        alpha = np.zeros((X.shape[0], K))
        for i in range(X.shape[0]):
            a, _ = nnls(Z.T, X[i])
            alpha[i] = project_to_simplex(a)

        # Variante nnls_alpha_beta_projected:
        # alpha se estima por NNLS y beta usa la actualizacion proyectada
        # existente para conservar la estructura actual del proyecto.
        beta = update_beta(alpha, X)

        Z = compute_archetypes(beta, X)
        X_hat = alpha @ Z
        error = np.linalg.norm(X - X_hat, 'fro')

        print(f"Iter {it:03d} | error: {error:.6f}")

        # criterio de convergencia
        if abs(prev_error - error) < 1e-6:
            print("Convergencia alcanzada")
            break

        prev_error = error

    return alpha, beta, Z


def project_to_simplex_euclidean(v):
    v = np.asarray(v, dtype=float)

    if v.ndim != 1:
        raise ValueError("project_to_simplex_euclidean espera un vector 1D")

    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - 1)
    if not np.any(rho_candidates):
        return np.full(n, 1.0 / n)

    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def project_rows_to_simplex_euclidean(M):
    M_proj = np.zeros_like(M, dtype=float)

    for i in range(M.shape[0]):
        M_proj[i] = project_to_simplex_euclidean(M[i])

    return M_proj


def reconstruction_error(X, alpha, beta):
    Z = compute_archetypes(beta, X)
    X_hat = alpha @ Z
    return np.linalg.norm(X - X_hat, "fro")


def update_alpha_pg_stable(X, Z, alpha, lr_alpha=LR_ALPHA, n_steps=ALPHA_ITER):
    for _ in range(n_steps):
        grad_alpha = 2 * (alpha @ Z - X) @ Z.T
        alpha = project_rows_to_simplex_euclidean(alpha - lr_alpha * grad_alpha)

    return alpha


def compute_weights(X, X_hat, c=1.5):
    residuals = np.linalg.norm(X - X_hat, axis=1)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median)) + 1e-8
    r = residuals / (1.4826 * mad)

    weights = np.ones_like(r)
    weights[r > c] = c / r[r > c]
    return weights


def update_beta_global_robust(
    X,
    alpha,
    beta,
    lr_beta=ROBUST_LR_BETA,
    n_steps=BETA_ITER,
):
    beta_candidate = beta.copy()

    for _ in range(n_steps):
        Z = compute_archetypes(beta_candidate, X)
        X_hat = alpha @ Z
        E = X - X_hat
        weights = compute_weights(X, X_hat)
        grad_beta = -2 * (alpha.T @ (weights[:, None] * E)) @ X.T
        grad_beta = grad_beta / X.shape[0]
        beta_candidate = project_rows_to_simplex_euclidean(
            beta_candidate - lr_beta * grad_beta
        )

    return beta_candidate


def init_random_beta_robust(X, K):
    n = X.shape[0]
    if K > n:
        raise ValueError(f"K ({K}) no puede ser mayor que n_samples ({n})")

    beta = np.zeros((K, n))
    indices = np.random.choice(n, K, replace=False)
    beta[np.arange(K), indices] = 1.0
    return beta


def init_pca_beta_robust(X, K):
    X = np.asarray(X)
    n = X.shape[0]

    if K > n:
        raise ValueError(f"K ({K}) no puede ser mayor que n_samples ({n})")

    coords = PCA(n_components=2).fit_transform(X)
    candidatos = [
        int(np.argmax(coords[:, 0])),
        int(np.argmin(coords[:, 0])),
        int(np.argmax(coords[:, 1])),
        int(np.argmin(coords[:, 1])),
    ]

    seleccionados = []
    for idx in candidatos:
        if idx not in seleccionados:
            seleccionados.append(idx)

    while len(seleccionados) < K:
        candidato = int(np.random.randint(0, n))
        if candidato not in seleccionados:
            seleccionados.append(candidato)

    seleccionados = seleccionados[:K]
    beta = np.zeros((K, n))
    beta[np.arange(K), seleccionados] = 1.0
    return beta


def inicializar_beta_robust(X, K, seed=42):
    from config import INICIALIZACION

    np.random.seed(seed)

    if INICIALIZACION == "random":
        return init_random_beta_robust(X, K)

    if INICIALIZACION == "pca":
        return init_pca_beta_robust(X, K)

    if INICIALIZACION == "furthest_sum":
        return inicializar_beta(X, K, seed=seed, metodo="furthest_sum")

    raise ValueError(f"Inicializacion no soportada para robust_pgd: {INICIALIZACION}")


def robust_pgd_archetypal_analysis(
    X,
    K,
    max_iter=None,
    seed=42,
    lr_alpha=LR_ALPHA,
    lr_beta=ROBUST_LR_BETA,
    alpha_iter=ALPHA_ITER,
    beta_iter=BETA_ITER,
    patience=PATIENCE,
):

    if max_iter is None:
        max_iter = MAX_ITER

    n = X.shape[0]

    beta = inicializar_beta_robust(X, K, seed=seed)
    beta = project_rows_to_simplex_euclidean(beta)
    alpha = np.random.dirichlet(np.ones(K), size=n)

    best_error = np.inf
    best_alpha = None
    best_beta = None
    best_Z = None
    errors = []
    no_improve = 0

    for it in range(max_iter):

        Z = compute_archetypes(beta, X)
        alpha = update_alpha_pg_stable(
            X,
            Z,
            alpha,
            lr_alpha=lr_alpha,
            n_steps=alpha_iter,
        )

        beta_candidate = update_beta_global_robust(
            X,
            alpha,
            beta,
            lr_beta=lr_beta,
            n_steps=beta_iter,
        )
        current_error = reconstruction_error(X, alpha, beta)
        error_candidate = reconstruction_error(X, alpha, beta_candidate)

        if error_candidate <= current_error * 1.001:
            beta = beta_candidate
            error = error_candidate
        else:
            error = current_error

        Z = compute_archetypes(beta, X)
        errors.append(float(error))

        print(f"Iter {it:03d} | error: {error:.6f}")

        if error < best_error:
            best_error = error
            best_alpha = alpha.copy()
            best_beta = beta.copy()
            best_Z = Z.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_alpha, best_beta, best_Z, errors

def evaluate_K_range(X, K_values, max_iter=50):

    errors = []

    for K in K_values:

        print(f"\nK = {K}")

        try:
            alpha, beta, Z = archetypal_analysis(X, K, max_iter=max_iter)
            err = np.linalg.norm(X - alpha @ Z, 'fro')

            errors.append(err)

        except Exception as e:
            print(f"Error K={K}: {e}")
            errors.append(np.nan)

    return errors


# Función principal para optimizar según la configuración
def optimizar(X, K, max_iter=None, seed=42):

    if OPTIMIZACION == "alternating":
        return archetypal_analysis(X, K, max_iter, seed)

    elif OPTIMIZACION == "pgd":
        return pgd_archetypal_analysis(X, K, max_iter, seed)

    elif OPTIMIZACION == "nnls":
        return nnls_alpha_beta_projected(X, K, max_iter, seed)

    elif OPTIMIZACION == "robust_pgd":
        return robust_pgd_archetypal_analysis(X, K, max_iter, seed)

    else:
        raise ValueError(
            f"Optimización no soportada: {OPTIMIZACION}"
        )
