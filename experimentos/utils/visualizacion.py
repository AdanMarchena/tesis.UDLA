"""
visualizacion.py

Figuras estandarizadas para experimentos de Analisis Arquetipico.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _asegurar_subcarpeta(carpeta_figuras, nombre: str) -> Path:
    carpeta = Path(carpeta_figuras) / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def _fase_duplicada(m: int) -> np.ndarray:
    fase = np.linspace(0, 1, m)
    return np.concatenate([fase, fase + 1])


def _curva_duplicada(curva) -> np.ndarray:
    curva = np.asarray(curva)
    return np.concatenate([curva, curva])


def plot_arquetipos_duplicados(Z, carpeta_figuras):
    """
    Guarda los arquetipos duplicando la fase de 0 a 2.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z debe tener shape (K, M). Shape recibido: {Z.shape}")

    carpeta = _asegurar_subcarpeta(carpeta_figuras, "arquetipos")
    ruta = carpeta / "arquetipos_duplicados.png"

    fase = _fase_duplicada(Z.shape[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, arquetipo in enumerate(Z):
        ax.plot(fase, _curva_duplicada(arquetipo), label=f"Arquetipo {k + 1}")

    ax.set_xlabel("Fase")
    ax.set_ylabel("Magnitud normalizada")
    ax.set_title("Arquetipos duplicados")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta


def plot_reconstrucciones(X, X_hat, ids, carpeta_figuras, indices=None):
    """
    Guarda curvas original vs reconstruccion para indices fijos.
    """
    X = np.asarray(X)
    X_hat = np.asarray(X_hat)
    ids = np.asarray(ids)

    if indices is None:
        indices = [0, 1, 2]

    carpeta = _asegurar_subcarpeta(carpeta_figuras, "reconstrucciones")
    fase = _fase_duplicada(X.shape[1])
    rutas = []

    for indice in indices:
        if indice < 0 or indice >= X.shape[0]:
            continue

        ruta = carpeta / f"reconstruccion_{indice:03d}.png"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fase, _curva_duplicada(X[indice]), label="Original", linewidth=2)
        ax.plot(
            fase,
            _curva_duplicada(X_hat[indice]),
            label="Reconstruccion",
            linewidth=2,
            linestyle="--",
        )

        ax.set_xlabel("Fase")
        ax.set_ylabel("Magnitud normalizada")
        ax.set_title(f"Reconstruccion {indice:03d} | {ids[indice]}")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)

        rutas.append(ruta)

    return rutas


def plot_simplex(alpha, carpeta_figuras):
    """
    Proyecta alpha con K=3 a coordenadas baricentricas 2D.
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or alpha.shape[1] != 3:
        print("plot_simplex omitido: alpha no tiene K=3.")
        return None

    carpeta = _asegurar_subcarpeta(carpeta_figuras, "proyecciones")
    ruta = carpeta / "simplex.png"

    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ]
    )
    coords = alpha @ vertices
    triangulo = np.vstack([vertices, vertices[0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(triangulo[:, 0], triangulo[:, 1], color="black", linewidth=1.5)
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7)
    ax.set_title("Simplex de coeficientes alpha")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta


def plot_alpha_pca(alpha, carpeta_figuras):
    """
    Proyecta alpha a 2D usando PCA de sklearn.
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or alpha.shape[1] <= 3:
        print("plot_alpha_pca omitido: usar solo cuando K > 3.")
        return None

    from sklearn.decomposition import PCA

    carpeta = _asegurar_subcarpeta(carpeta_figuras, "proyecciones")
    ruta = carpeta / "alpha_pca.png"

    coords = PCA(n_components=2).fit_transform(alpha)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Proyeccion PCA de alpha")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta


def plot_codo(K_values, errors, carpeta_figuras):
    """
    Guarda grafico de codo para diagnostico de K.
    """
    carpeta = _asegurar_subcarpeta(carpeta_figuras, "diagnostico")
    ruta = carpeta / "codo.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_values, errors, marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Error")
    ax.set_title("Diagnostico de codo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta
