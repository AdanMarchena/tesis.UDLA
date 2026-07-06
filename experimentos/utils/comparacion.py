"""
comparacion.py

Herramientas para comparar experimentos AA ya exportados.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR
from utils.loaders import cargar_dataset
import utils.loaders as loaders
from utils.normalizacion import minmax, zscore
from utils.suavizado import savgol, sin_suavizado


def _nombre_seguro(nombre: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nombre)


def _resolver_experimento(exp) -> Path:
    ruta = Path(exp)

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

    raise FileNotFoundError(f"No se encontro el experimento: {exp}")


def _cargar_json(ruta: Path) -> dict:
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def _cargar_metricas(carpeta_exp: Path) -> dict:
    candidatos = [
        carpeta_exp / "metricas" / "metricas.json",
        carpeta_exp / "metricas" / "metrics.json",
    ]
    for ruta in candidatos:
        if ruta.exists():
            return _cargar_json(ruta)
    raise FileNotFoundError(
        "No se encontro metricas/metricas.json ni metricas/metrics.json "
        f"en {carpeta_exp}"
    )


def _cargar_modelo(carpeta_exp: Path) -> dict:
    carpeta_modelos = carpeta_exp / "modelos"
    modelo = {}
    for nombre in ("alpha", "beta", "Z", "X_hat"):
        ruta = carpeta_modelos / f"{nombre}.npy"
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}")
        modelo[nombre] = np.load(ruta)
    return modelo


def _fase_duplicada(m: int) -> np.ndarray:
    fase = np.linspace(0, 1, m)
    return np.concatenate([fase, fase + 1])


def _curva_duplicada(curva) -> np.ndarray:
    curva = np.asarray(curva)
    return np.concatenate([curva, curva])


def _normalizar_desde_config(X, cfg: dict):
    normalizacion = cfg.get("normalizacion", "minmax")
    if normalizacion == "minmax":
        return minmax(X)
    if normalizacion == "zscore":
        return zscore(X)
    raise ValueError(f"Normalizacion no soportada: {normalizacion}")


def _suavizar_desde_config(X, cfg: dict):
    suavizado = cfg.get("suavizado", "none")
    if suavizado == "none":
        return sin_suavizado(X)
    if suavizado == "savgol":
        return savgol(
            X,
            window=cfg.get("savgol_window", cfg.get("SAVGOL_WINDOW", 5)),
            poly=cfg.get("savgol_poly", cfg.get("SAVGOL_POLY", 2)),
        )
    raise ValueError(f"Suavizado no soportado: {suavizado}")


def _cargar_X_original(cfg: dict):
    valores_originales = {
        "DATASET": loaders.DATASET,
        "DISCRETIZACION": loaders.DISCRETIZACION,
        "N_PUNTOS": loaders.N_PUNTOS,
    }

    try:
        loaders.DATASET = cfg.get("dataset")
        loaders.DISCRETIZACION = cfg.get("discretizacion")
        loaders.N_PUNTOS = cfg.get("n_puntos")
        X, ids = cargar_dataset(
            n_muestras=cfg.get("n_muestras"),
            seed=cfg.get("seed", 42),
        )
    finally:
        loaders.DATASET = valores_originales["DATASET"]
        loaders.DISCRETIZACION = valores_originales["DISCRETIZACION"]
        loaders.N_PUNTOS = valores_originales["N_PUNTOS"]

    X = _normalizar_desde_config(X, cfg)
    X = _suavizar_desde_config(X, cfg)
    return X, ids


def _config_datos(cfg: dict) -> dict:
    claves = (
        "dataset",
        "discretizacion",
        "n_puntos",
        "n_muestras",
        "normalizacion",
        "suavizado",
        "seed",
    )
    return {clave: cfg.get(clave) for clave in claves}


def _cargar_historial(carpeta_exp: Path):
    ruta_npy = carpeta_exp / "modelos" / "historial_error.npy"
    if ruta_npy.exists():
        return np.load(ruta_npy).astype(float).tolist()

    candidatos_json = [
        carpeta_exp / "metricas" / "historial_error.json",
        carpeta_exp / "metricas" / "metricas.json",
        carpeta_exp / "metricas" / "metrics.json",
    ]
    for ruta in candidatos_json:
        if not ruta.exists():
            continue
        datos = _cargar_json(ruta)
        if isinstance(datos, dict) and "historial_error" in datos:
            return [float(valor) for valor in datos["historial_error"]]

    print(f"historial_error no disponible para este experimento: {carpeta_exp.name}")
    return None


def _cargar_experimento(exp) -> dict:
    carpeta = _resolver_experimento(exp)
    config = _cargar_json(carpeta / "config" / "config.json")
    metricas = _cargar_metricas(carpeta)
    modelo = _cargar_modelo(carpeta)

    return {
        "nombre": carpeta.name,
        "carpeta": carpeta,
        "config": config,
        "metricas": metricas,
        "historial_error": _cargar_historial(carpeta),
        **modelo,
    }


def _tabla_resumen(exp_a: dict, exp_b: dict) -> pd.DataFrame:
    filas = []
    for exp in (exp_a, exp_b):
        cfg = exp["config"]
        metricas = exp["metricas"]
        filas.append(
            {
                "nombre_experimento": exp["nombre"],
                "optimizacion": cfg.get("optimizacion"),
                "inicializacion": cfg.get("inicializacion"),
                "K": cfg.get("K"),
                "error_rel": metricas.get("error_rel"),
                "error_fro": metricas.get("error_fro"),
                "convergio": cfg.get("convergio"),
                "iteraciones_reales": cfg.get("iteraciones_reales"),
                "tiempo_segundos": cfg.get("tiempo_segundos"),
            }
        )

    resumen = pd.DataFrame(filas)

    resumen_maestro = RESULTS_DIR / "resumen_experimentos.csv"
    if resumen_maestro.exists():
        maestro = pd.read_csv(resumen_maestro)
        for idx, exp in enumerate((exp_a, exp_b)):
            fila = maestro[
                maestro["ruta_experimento"].astype(str)
                == str(exp["carpeta"].relative_to(PROJECT_ROOT))
            ]
            if not fila.empty:
                fila = fila.tail(1).iloc[0]
                for columna in (
                    "convergio",
                    "iteraciones_reales",
                    "tiempo_segundos",
                ):
                    resumen.loc[idx, columna] = fila.get(columna)

    print("\nResumen comparativo:")
    print(resumen.to_string(index=False))
    return resumen


def _plot_arquetipos(exp_a: dict, exp_b: dict, carpeta_salida: Path) -> Path:
    ruta = carpeta_salida / "arquetipos_comparacion.png"
    Z_a = np.asarray(exp_a["Z"])
    Z_b = np.asarray(exp_b["Z"])
    k_max = max(Z_a.shape[0], Z_b.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, exp, Z in ((axes[0], exp_a, Z_a), (axes[1], exp_b, Z_b)):
        fase = _fase_duplicada(Z.shape[1])
        for k, arquetipo in enumerate(Z):
            ax.plot(fase, _curva_duplicada(arquetipo), label=f"A{k + 1}")
        ax.set_title(exp["nombre"])
        ax.set_xlabel("Fase")
        ax.set_ylabel("Magnitud normalizada")
        ax.invert_yaxis()
        if Z.shape[0] <= 12:
            ax.legend(ncol=min(k_max, 5), fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def _plot_reconstrucciones(
    exp_a: dict,
    exp_b: dict,
    X_a,
    X_b,
    ids_a,
    ids_b,
    indices,
    carpeta_salida: Path,
) -> Path:
    ruta = carpeta_salida / "reconstrucciones_comparacion.png"
    indices_validos = [
        indice
        for indice in indices
        if 0 <= indice < X_a.shape[0] and 0 <= indice < X_b.shape[0]
    ]
    if not indices_validos:
        raise ValueError("No hay indices de reconstruccion validos para ambos experimentos")

    fig, axes = plt.subplots(
        len(indices_validos),
        2,
        figsize=(12, 3.6 * len(indices_validos)),
        squeeze=False,
    )

    for fila, indice in enumerate(indices_validos):
        for col, (exp, X, ids) in enumerate(
            ((exp_a, X_a, ids_a), (exp_b, X_b, ids_b))
        ):
            ax = axes[fila, col]
            fase = _fase_duplicada(X.shape[1])
            ax.plot(fase, _curva_duplicada(X[indice]), label="Original", linewidth=2)
            ax.plot(
                fase,
                _curva_duplicada(exp["X_hat"][indice]),
                label="Reconstruccion",
                linewidth=2,
                linestyle="--",
            )
            ax.set_title(f"{exp['nombre']} | idx {indice} | {ids[indice]}")
            ax.set_xlabel("Fase")
            ax.set_ylabel("Magnitud normalizada")
            ax.invert_yaxis()
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def _simplex_coords(alpha):
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ]
    )
    return alpha @ vertices, np.vstack([vertices, vertices[0]])


def _plot_proyeccion(exp_a: dict, exp_b: dict, carpeta_salida: Path) -> Path:
    ruta = carpeta_salida / "proyeccion_comparacion.png"
    alpha_a = np.asarray(exp_a["alpha"])
    alpha_b = np.asarray(exp_b["alpha"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    if alpha_a.shape[1] == 3 and alpha_b.shape[1] == 3:
        for ax, exp, alpha in ((axes[0], exp_a, alpha_a), (axes[1], exp_b, alpha_b)):
            coords, triangulo = _simplex_coords(alpha)
            ax.plot(triangulo[:, 0], triangulo[:, 1], color="black", linewidth=1.5)
            ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.65)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.set_title(f"Simplex | {exp['nombre']}")
    else:
        from sklearn.decomposition import PCA

        for ax, exp, alpha in ((axes[0], exp_a, alpha_a), (axes[1], exp_b, alpha_b)):
            coords = PCA(n_components=2).fit_transform(alpha)
            ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.65)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA alpha | {exp['nombre']}")

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def _plot_convergencia(exp_a: dict, exp_b: dict, carpeta_salida: Path) -> Path | None:
    historial_a = exp_a["historial_error"]
    historial_b = exp_b["historial_error"]
    if historial_a is None and historial_b is None:
        return None

    ruta = carpeta_salida / "convergencia_comparacion.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if historial_a is not None:
        ax.plot(historial_a, label=exp_a["nombre"])
    if historial_b is not None:
        ax.plot(historial_b, label=exp_b["nombre"])
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Error")
    ax.set_title("Convergencia")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def comparar_experimentos(exp_a, exp_b, indices_reconstruccion=[0, 50, 999]):
    exp_a = _cargar_experimento(exp_a)
    exp_b = _cargar_experimento(exp_b)

    nombre_comparacion = (
        f"{_nombre_seguro(exp_a['nombre'])}_VS_{_nombre_seguro(exp_b['nombre'])}"
    )
    carpeta_salida = RESULTS_DIR / "comparaciones" / nombre_comparacion
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    resumen = _tabla_resumen(exp_a, exp_b)

    misma_config_datos = _config_datos(exp_a["config"]) == _config_datos(exp_b["config"])
    X_a, ids_a = _cargar_X_original(exp_a["config"])
    if misma_config_datos:
        X_b, ids_b = X_a, ids_a
    else:
        warnings.warn(
            "Las configuraciones de datos son distintas; se cargara X por separado.",
            stacklevel=2,
        )
        X_b, ids_b = _cargar_X_original(exp_b["config"])

    rutas = {
        "arquetipos": _plot_arquetipos(exp_a, exp_b, carpeta_salida),
        "reconstrucciones": _plot_reconstrucciones(
            exp_a,
            exp_b,
            X_a,
            X_b,
            ids_a,
            ids_b,
            indices_reconstruccion,
            carpeta_salida,
        ),
        "proyeccion": _plot_proyeccion(exp_a, exp_b, carpeta_salida),
        "convergencia": _plot_convergencia(exp_a, exp_b, carpeta_salida),
    }

    return {
        "carpeta_comparacion": carpeta_salida,
        "resumen": resumen,
        "figuras": rutas,
    }
