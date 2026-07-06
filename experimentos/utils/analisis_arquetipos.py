"""
analisis_arquetipos.py

Herramientas de analisis de uso de arquetipos para experimentos AA exportados.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR


def _resolver_experimento(ruta_experimento) -> Path:
    ruta = Path(ruta_experimento)

    candidatos = []
    if ruta.is_absolute():
        candidatos.append(ruta)
    else:
        candidatos.extend(
            [
                PROJECT_ROOT / ruta,
                RESULTS_DIR / "experimentos" / ruta,
                ruta,
            ]
        )

    for candidato in candidatos:
        if candidato.exists():
            return candidato.resolve()

    raise FileNotFoundError(f"No se encontro el experimento: {ruta_experimento}")


def _cargar_config(carpeta_exp: Path) -> dict:
    ruta = carpeta_exp / "config" / "config.json"
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def analizar_uso_arquetipos(ruta_experimento) -> pd.DataFrame:
    carpeta_exp = _resolver_experimento(ruta_experimento)
    ruta_alpha = carpeta_exp / "modelos" / "alpha.npy"

    if not ruta_alpha.exists():
        raise FileNotFoundError(f"No existe {ruta_alpha}")

    alpha = np.load(ruta_alpha)
    if alpha.ndim != 2:
        raise ValueError(f"alpha debe tener shape (n_estrellas, K). Shape: {alpha.shape}")

    n_estrellas, K = alpha.shape
    dominantes = np.argmax(alpha, axis=1)
    dominant_count = np.bincount(dominantes, minlength=K)

    return pd.DataFrame(
        {
            "arquetipo": np.arange(1, K + 1),
            "mean_activation": alpha.mean(axis=0),
            "dominant_count": dominant_count.astype(int),
            "dominant_fraction": dominant_count / n_estrellas,
        }
    )


def comparar_uso_arquetipos(rutas_experimentos, guardar=True) -> pd.DataFrame:
    frames = []

    for ruta_experimento in rutas_experimentos:
        carpeta_exp = _resolver_experimento(ruta_experimento)
        config = _cargar_config(carpeta_exp)
        uso = analizar_uso_arquetipos(carpeta_exp)

        uso.insert(0, "experimento", carpeta_exp.name)
        uso["K"] = config.get("K")
        uso["optimizacion"] = config.get("optimizacion")
        uso["discretizacion"] = config.get("discretizacion")
        uso["n_puntos"] = config.get("n_puntos")
        frames.append(uso)

    if frames:
        df_uso = pd.concat(frames, ignore_index=True)
    else:
        df_uso = pd.DataFrame(
            columns=[
                "experimento",
                "arquetipo",
                "mean_activation",
                "dominant_count",
                "dominant_fraction",
                "K",
                "optimizacion",
                "discretizacion",
                "n_puntos",
            ]
        )

    if guardar:
        carpeta = RESULTS_DIR / "analisis"
        carpeta.mkdir(parents=True, exist_ok=True)
        df_uso.to_csv(carpeta / "uso_arquetipos.csv", index=False)

    return df_uso


def plot_uso_arquetipos(df_uso) -> dict[str, Path]:
    df_uso = pd.DataFrame(df_uso).copy()
    if df_uso.empty:
        raise ValueError("df_uso esta vacio")

    carpeta = RESULTS_DIR / "analisis"
    carpeta.mkdir(parents=True, exist_ok=True)

    rutas = {
        "mean_activation": carpeta / "uso_arquetipos_mean_activation.png",
        "dominant_count": carpeta / "uso_arquetipos_dominant_count.png",
    }

    etiquetas = [
        f"K={int(row.K)} A{int(row.arquetipo)}"
        for row in df_uso.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df_uso)), 5))
    ax.bar(etiquetas, df_uso["mean_activation"].values)
    ax.set_xlabel("Arquetipo")
    ax.set_ylabel("Mean activation")
    ax.set_title("Activacion media por arquetipo")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(rutas["mean_activation"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df_uso)), 5))
    ax.bar(etiquetas, df_uso["dominant_count"].values)
    ax.set_xlabel("Arquetipo")
    ax.set_ylabel("Dominant count")
    ax.set_title("Estrellas dominadas por arquetipo")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(rutas["dominant_count"], dpi=150)
    plt.close(fig)

    return rutas


def analizar_similitud_arquetipos(ruta_experimento) -> dict:
    carpeta_exp = _resolver_experimento(ruta_experimento)
    ruta_z = carpeta_exp / "modelos" / "Z.npy"

    if not ruta_z.exists():
        raise FileNotFoundError(f"No existe {ruta_z}")

    Z = np.load(ruta_z)
    if Z.ndim != 2:
        raise ValueError(f"Z debe tener shape (K, M). Shape: {Z.shape}")

    K = Z.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        matriz_correlacion = np.corrcoef(Z)
    diferencias = Z[:, None, :] - Z[None, :, :]
    matriz_distancias = np.linalg.norm(diferencias, axis=2)
    mascara_offdiag = ~np.eye(K, dtype=bool)

    corr_offdiag = matriz_correlacion[mascara_offdiag]
    dist_offdiag = matriz_distancias[mascara_offdiag]
    corr_offdiag_valida = corr_offdiag[~np.isnan(corr_offdiag)]

    pares_muy_similares = []
    for i in range(K):
        for j in range(i + 1, K):
            if not np.isnan(matriz_correlacion[i, j]) and matriz_correlacion[i, j] > 0.98:
                pares_muy_similares.append(
                    {
                        "arquetipo_i": i + 1,
                        "arquetipo_j": j + 1,
                        "correlacion": float(matriz_correlacion[i, j]),
                        "distancia": float(matriz_distancias[i, j]),
                    }
                )

    resumen = {
        "max_corr_offdiag": float(np.max(corr_offdiag_valida))
        if corr_offdiag_valida.size
        else np.nan,
        "mean_corr_offdiag": float(np.mean(corr_offdiag_valida))
        if corr_offdiag_valida.size
        else np.nan,
        "min_dist_offdiag": float(np.min(dist_offdiag)) if dist_offdiag.size else np.nan,
        "mean_dist_offdiag": float(np.mean(dist_offdiag)) if dist_offdiag.size else np.nan,
        "pares_muy_similares": pares_muy_similares,
    }

    return {
        "matriz_correlacion": matriz_correlacion,
        "matriz_distancias": matriz_distancias,
        "resumen": resumen,
    }


def comparar_similitud_arquetipos(rutas_experimentos, guardar=True) -> pd.DataFrame:
    filas = []

    for ruta_experimento in rutas_experimentos:
        carpeta_exp = _resolver_experimento(ruta_experimento)
        config = _cargar_config(carpeta_exp)
        resultado = analizar_similitud_arquetipos(carpeta_exp)
        resumen = resultado["resumen"]

        filas.append(
            {
                "experimento": carpeta_exp.name,
                "K": config.get("K"),
                "optimizacion": config.get("optimizacion"),
                "discretizacion": config.get("discretizacion"),
                "n_puntos": config.get("n_puntos"),
                "max_corr_offdiag": resumen["max_corr_offdiag"],
                "mean_corr_offdiag": resumen["mean_corr_offdiag"],
                "min_dist_offdiag": resumen["min_dist_offdiag"],
                "mean_dist_offdiag": resumen["mean_dist_offdiag"],
                "n_pares_muy_similares": len(resumen["pares_muy_similares"]),
            }
        )

    df_similitud = pd.DataFrame(filas)

    if guardar:
        carpeta = RESULTS_DIR / "analisis"
        carpeta.mkdir(parents=True, exist_ok=True)
        df_similitud.to_csv(carpeta / "similitud_arquetipos.csv", index=False)

    return df_similitud


def plot_similitud_arquetipos(ruta_experimento) -> dict[str, Path]:
    carpeta_exp = _resolver_experimento(ruta_experimento)
    resultado = analizar_similitud_arquetipos(carpeta_exp)
    correlacion = resultado["matriz_correlacion"]
    distancias = resultado["matriz_distancias"]

    carpeta = RESULTS_DIR / "analisis"
    carpeta.mkdir(parents=True, exist_ok=True)

    rutas = {
        "correlacion": carpeta
        / f"similitud_{carpeta_exp.name}_correlacion.png",
        "distancia": carpeta
        / f"similitud_{carpeta_exp.name}_distancia.png",
    }

    etiquetas = [f"A{i + 1}" for i in range(correlacion.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlacion, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(etiquetas)), labels=etiquetas)
    ax.set_yticks(np.arange(len(etiquetas)), labels=etiquetas)
    ax.set_title(f"Correlacion entre arquetipos | {carpeta_exp.name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(rutas["correlacion"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(distancias, cmap="viridis")
    ax.set_xticks(np.arange(len(etiquetas)), labels=etiquetas)
    ax.set_yticks(np.arange(len(etiquetas)), labels=etiquetas)
    ax.set_title(f"Distancia euclidiana entre arquetipos | {carpeta_exp.name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(rutas["distancia"], dpi=150)
    plt.close(fig)

    return rutas
