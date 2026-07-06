"""
exportacion.py

Gestion completa de experimentos:
- config
- metricas
- resultados del modelo
"""

from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime
import os

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR


SUBCARPETAS = (
    "config",
    "metricas",
    "modelos",
    "figuras",
    "figuras/arquetipos",
    "figuras/reconstrucciones",
    "figuras/proyecciones",
    "figuras/diagnostico",
)


# ============================================================
# IDENTIDAD DEL EXPERIMENTO
# ============================================================

def _normalizar_valor(valor):
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, dict):
        return {str(k): _normalizar_valor(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_normalizar_valor(v) for v in valor]
    return valor


def config_serializable(config: dict) -> dict:
    return _normalizar_valor(config)


def generar_id(config: dict) -> str:
    texto = json.dumps(config_serializable(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:10].upper()


def ruta_relativa_proyecto(ruta) -> str:
    ruta = Path(ruta).resolve()

    try:
        return str(ruta.relative_to(PROJECT_ROOT))
    except ValueError:
        return os.path.relpath(ruta, PROJECT_ROOT)


def ruta_experimento(config: dict, sufijo: str | None = None) -> Path:
    exp_id = generar_id(config)

    nombre = (
        f"{config['modelo']}_"
        f"{exp_id}_"
        f"{config['dataset']}_"
        f"{config['discretizacion']}_"
        f"{config['n_puntos']}_"
        f"K{config['K']}"
    )

    if sufijo:
        nombre = f"{nombre}_{sufijo}"

    return RESULTS_DIR / "experimentos" / nombre


# ============================================================
# CREACION
# ============================================================

def existe_experimento(config: dict) -> bool:
    return ruta_experimento(config).exists()


def _crear_subcarpetas(carpeta: Path) -> None:
    for subcarpeta in SUBCARPETAS:
        (carpeta / subcarpeta).mkdir(parents=True, exist_ok=True)


def crear_experimento(config: dict, carpeta: Path | None = None) -> Path:
    carpeta = carpeta or ruta_experimento(config)
    _crear_subcarpetas(carpeta)
    return carpeta


def _ruta_nueva(config: dict) -> Path:
    base = ruta_experimento(config)
    if not base.exists():
        return base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidato = ruta_experimento(config, sufijo=f"new_{timestamp}")

    contador = 1
    while candidato.exists():
        candidato = ruta_experimento(config, sufijo=f"new_{timestamp}_{contador:02d}")
        contador += 1

    return candidato


def preparar_experimento(config: dict, modo_ejecucion: str = "skip") -> tuple[Path, bool]:
    modo = modo_ejecucion.lower()
    carpeta = ruta_experimento(config)

    if modo not in {"skip", "overwrite", "new"}:
        raise ValueError(f"Modo de ejecucion no soportado: {modo_ejecucion}")

    if modo == "skip" and carpeta.exists():
        return carpeta, False

    if modo == "overwrite" and carpeta.exists():
        shutil.rmtree(carpeta)

    if modo == "new":
        carpeta = _ruta_nueva(config)

    crear_experimento(config, carpeta)
    return carpeta, True


# ============================================================
# GUARDADO
# ============================================================

def guardar_config(config: dict, carpeta: Path | None = None) -> Path:
    carpeta = carpeta or ruta_experimento(config)
    ruta = carpeta / "config" / "config.json"

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(config_serializable(config), f, indent=4, ensure_ascii=False)

    return ruta


def guardar_metricas(config: dict, metrics: dict, carpeta: Path | None = None) -> Path:
    carpeta = carpeta or ruta_experimento(config)
    ruta = carpeta / "metricas" / "metricas.json"

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_normalizar_valor(metrics), f, indent=4, ensure_ascii=False)

    return ruta


def guardar_modelo(
    config: dict,
    alpha,
    Z,
    X_hat,
    beta=None,
    carpeta: Path | None = None,
) -> list[Path]:
    carpeta = carpeta or ruta_experimento(config)
    carpeta_modelos = carpeta / "modelos"

    archivos = []
    for nombre, matriz in {
        "alpha.npy": alpha,
        "beta.npy": beta,
        "Z.npy": Z,
        "X_hat.npy": X_hat,
    }.items():
        if matriz is None:
            continue

        ruta = carpeta_modelos / nombre
        np.save(ruta, matriz)
        archivos.append(ruta)

    return archivos


# ============================================================
# RESUMEN MAESTRO
# ============================================================

def _metricas_resumen(metrics: dict) -> dict:
    error_per_sample = np.asarray(metrics.get("error_per_sample", []), dtype=float)

    if error_per_sample.size == 0:
        return {
            "error_per_sample_mean": np.nan,
            "error_per_sample_std": np.nan,
            "error_per_sample_min": np.nan,
            "error_per_sample_max": np.nan,
        }

    return {
        "error_per_sample_mean": float(np.mean(error_per_sample)),
        "error_per_sample_std": float(np.std(error_per_sample)),
        "error_per_sample_min": float(np.min(error_per_sample)),
        "error_per_sample_max": float(np.max(error_per_sample)),
    }


def _shape_resumen(X) -> tuple[int | None, int | None]:
    if X is None:
        return None, None

    X = np.asarray(X)
    if X.ndim == 0:
        return None, None

    n_estrellas = int(X.shape[0])
    n_variables = int(X.shape[1]) if X.ndim > 1 else 1
    return n_estrellas, n_variables


def actualizar_resumen_experimentos(
    config: dict,
    metrics: dict,
    carpeta_exp,
    tiempo_segundos,
    X,
    metadata: dict | None = None,
) -> dict:
    carpeta_exp = Path(carpeta_exp)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    n_estrellas, n_variables = _shape_resumen(X)

    iteraciones_reales = metadata.get("iteraciones_reales")
    tiempo_por_iteracion = metadata.get("tiempo_por_iteracion")
    if tiempo_por_iteracion is None and iteraciones_reales:
        tiempo_por_iteracion = float(tiempo_segundos) / iteraciones_reales

    loss_final = metadata.get("loss_final", metrics.get("error_fro", np.nan))

    fila = {
        "experiment_id": generar_id(config),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": config.get("modelo"),
        "dataset": config.get("dataset"),
        "discretizacion": config.get("discretizacion"),
        "n_puntos": config.get("n_puntos"),
        "n_muestras": config.get("n_muestras"),
        "normalizacion": config.get("normalizacion"),
        "suavizado": config.get("suavizado") or "none",
        "inicializacion": config.get("inicializacion"),
        "optimizacion": config.get("optimizacion"),
        "K": config.get("K"),
        "max_iter": config.get("max_iter"),
        "seed": config.get("seed"),
        "n_estrellas": n_estrellas,
        "n_variables": n_variables,
        "error_fro": float(metrics.get("error_fro", np.nan)),
        "error_rel": float(metrics.get("error_rel", np.nan)),
        **_metricas_resumen(metrics),
        "tiempo_segundos": float(tiempo_segundos),
        "estado": metadata.get("estado", "OK"),
        "convergio": metadata.get("convergio"),
        "iteraciones_reales": iteraciones_reales,
        "tiempo_por_iteracion": tiempo_por_iteracion,
        "version_algoritmo": metadata.get("version_algoritmo"),
        "loss_final": float(loss_final) if loss_final is not None else np.nan,
        "purity": metadata.get("purity", np.nan),
        "mean_activation": metadata.get("mean_activation", np.nan),
        "silhouette": metadata.get("silhouette", np.nan),
        "davies_bouldin": metadata.get("davies_bouldin", np.nan),
        "experimento_final": bool(metadata.get("experimento_final", False)),
        "ruta_experimento": ruta_relativa_proyecto(carpeta_exp),
    }

    ruta_parquet = RESULTS_DIR / "resumen_experimentos.parquet"
    ruta_csv = RESULTS_DIR / "resumen_experimentos.csv"

    if ruta_csv.exists():
        resumen = pd.read_csv(ruta_csv)
        if "experimento_final" not in resumen.columns:
            resumen["experimento_final"] = False
        resumen = pd.concat([resumen, pd.DataFrame([fila])], ignore_index=True)
    else:
        resumen = pd.DataFrame([fila])

    try:
        resumen.to_parquet(ruta_parquet, index=False)
        parquet_actualizado = True
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        parquet_actualizado = False
        print(
            "No se pudo guardar resumen_experimentos.parquet "
            f"({exc}). Se mantiene resumen_experimentos.csv."
        )

    resumen.to_csv(ruta_csv, index=False)

    return {
        "parquet": ruta_parquet,
        "csv": ruta_csv,
        "parquet_actualizado": parquet_actualizado,
        "csv_actualizado": True,
    }


# ============================================================
# LIMPIEZA OPCIONAL
# ============================================================

def eliminar_experimento(config: dict) -> None:
    carpeta = ruta_experimento(config)

    if carpeta.exists():
        shutil.rmtree(carpeta)
