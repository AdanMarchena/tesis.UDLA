"""
recalcular_metricas.py

Auditoria y recálculo de métricas de agrupamiento/pureza para experimentos
ya ejecutados. No reentrena modelos.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.optimize import linear_sum_assignment

from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from utils.suavizado import savgol


SAMPLE_SIZE_SILHOUETTE = 10000
RANDOM_STATE_METRICAS = 42


def resolver_ruta(ruta) -> Path:
    ruta = Path(str(ruta))
    candidatos = [ruta]
    if not ruta.is_absolute():
        candidatos = [
            PROJECT_ROOT / ruta,
            RESULTS_DIR / "experimentos" / ruta,
            ruta,
        ]

    for candidato in candidatos:
        if candidato.exists():
            return candidato.resolve()

    raise FileNotFoundError(f"No se encontro la ruta: {ruta}")


def cargar_config_experimento(carpeta: Path, fila: pd.Series | None = None) -> dict:
    posibles = [
        carpeta / "config" / "config.json",
        carpeta / "config.json",
    ]
    for ruta in posibles:
        if ruta.exists():
            with ruta.open("r", encoding="utf-8") as archivo:
                return json.load(archivo)

    if fila is None:
        raise FileNotFoundError(f"No existe config.json en {carpeta}")

    def valor(columna):
        salida = fila.get(columna)
        if pd.isna(salida):
            return None
        return salida

    return {
        "modelo": valor("modelo"),
        "dataset": valor("dataset"),
        "discretizacion": valor("discretizacion"),
        "n_puntos": int(valor("n_puntos")),
        "n_muestras": None if valor("n_muestras") is None else int(valor("n_muestras")),
        "normalizacion": valor("normalizacion"),
        "suavizado": valor("suavizado") or "none",
        "inicializacion": valor("inicializacion"),
        "optimizacion": valor("optimizacion"),
        "K": int(valor("K")),
        "max_iter": valor("max_iter"),
        "seed": int(valor("seed")),
    }


def cargar_ids_guardados(carpeta: Path):
    candidatos = [
        carpeta / "modelos" / "ids.npy",
        carpeta / "ids.npy",
        carpeta / "modelos" / "star_ids.npy",
        carpeta / "star_ids.npy",
    ]
    for ruta in candidatos:
        if ruta.exists():
            return np.load(ruta, allow_pickle=True).astype(str), ruta
    return None, None


def normalizar_X(X, metodo: str):
    X = np.asarray(X, dtype=float)
    if metodo == "minmax":
        minimo = np.min(X, axis=1, keepdims=True)
        maximo = np.max(X, axis=1, keepdims=True)
        return (X - minimo) / (maximo - minimo + 1e-8)
    if metodo == "zscore":
        media = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - media) / (std + 1e-8)
    raise ValueError(f"Normalizacion no soportada: {metodo}")


def suavizar_X(X, config: dict):
    metodo = config.get("suavizado") or "none"
    if metodo == "none":
        return X
    if metodo == "savgol":
        window = int(config.get("SAVGOL_WINDOW", config.get("savgol_window", 5)))
        poly = int(config.get("SAVGOL_POLY", config.get("savgol_poly", 2)))
        return savgol(X, window=window, poly=poly)
    raise ValueError(f"Suavizado no soportado: {metodo}")


def cargar_X_experimento(config: dict, ids=None):
    ruta = DATA_DIR / config["discretizacion"] / f"{config['dataset']}_{int(config['n_puntos'])}.parquet"
    df = pd.read_parquet(ruta)
    ids_base = df["star_id"].astype(str).to_numpy()
    X = df.drop(columns=["star_id"]).to_numpy()

    if ids is not None:
        posiciones = pd.Series(np.arange(len(ids_base)), index=ids_base)
        if len(set(ids)) != len(ids):
            raise ValueError("IDs guardados contienen duplicados")
        if not set(ids).issubset(set(posiciones.index)):
            raise ValueError("Hay IDs guardados que no existen en el dataset fuente")
        idx = posiciones.loc[list(ids)].to_numpy()
        ids_out = ids_base[idx]
        X = X[idx]
    else:
        n_muestras = config.get("n_muestras")
        if n_muestras is None:
            idx = np.arange(len(ids_base))
        else:
            rng = np.random.default_rng(int(config.get("seed", 42)))
            idx = rng.choice(len(ids_base), size=int(n_muestras), replace=False)
        ids_out = ids_base[idx]
        X = X[idx]

    X = normalizar_X(X, config.get("normalizacion"))
    X = suavizar_X(X, config)
    return X, ids_out


def obtener_labels_evaluacion(modelo: str, carpeta: Path):
    modelo = str(modelo).upper()
    if modelo == "AA":
        ruta_alpha = carpeta / "modelos" / "alpha.npy"
        if not ruta_alpha.exists():
            raise FileNotFoundError(f"No existe {ruta_alpha}")
        alpha = np.load(ruta_alpha)
        return np.argmax(alpha, axis=1), {"alpha": alpha}

    if modelo == "KMEANS":
        ruta_labels = carpeta / "labels.npy"
        if not ruta_labels.exists():
            raise FileNotFoundError(f"No existe {ruta_labels}")
        labels = np.load(ruta_labels)
        return labels.astype(int), {}

    if modelo == "ADA":
        ruta_asignaciones = carpeta / "asignaciones.npy"
        if not ruta_asignaciones.exists():
            raise FileNotFoundError(f"No existe {ruta_asignaciones}")
        asignaciones = np.load(ruta_asignaciones)
        return asignaciones.astype(int), {}

    raise ValueError(f"Modelo no soportado para labels de evaluacion: {modelo}")


def star_id_base(star_id):
    texto = str(star_id)
    for prefijo in ("OGLEIII_", "OGLEIV_"):
        if texto.startswith(prefijo):
            return texto[len(prefijo) :]
    return texto


def cargar_subtipos(ids):
    catalogo = pd.read_csv(RESULTS_DIR / "catalogos" / "rrlyrae_subtipos.csv")
    catalogo = catalogo[["star_id", "star_id_base", "subtipo_rrlyrae"]].drop_duplicates()
    tabla = pd.DataFrame({"star_id": np.asarray(ids, dtype=str)})
    tabla["star_id_base"] = tabla["star_id"].map(star_id_base)
    tabla = tabla.merge(
        catalogo.drop(columns=["star_id_base"]),
        on="star_id",
        how="left",
    )
    faltantes = tabla["subtipo_rrlyrae"].isna()
    if faltantes.any():
        catalogo_base = catalogo[["star_id_base", "subtipo_rrlyrae"]].drop_duplicates("star_id_base")
        relleno = tabla.loc[faltantes, ["star_id_base"]].merge(catalogo_base, on="star_id_base", how="left")
        tabla.loc[faltantes, "subtipo_rrlyrae"] = relleno["subtipo_rrlyrae"].to_numpy()
    return tabla["subtipo_rrlyrae"].to_numpy()


def calcular_pureza(labels, subtipos):
    labels = np.asarray(labels)
    subtipos = np.asarray(subtipos)
    validos = pd.notna(subtipos)
    if validos.sum() == 0:
        raise ValueError("No hay subtipos disponibles para calcular pureza")

    df = pd.DataFrame({"label": labels[validos], "subtipo": subtipos[validos]})
    conteos = pd.crosstab(df["label"], df["subtipo"])
    purity = float(conteos.max(axis=1).sum() / conteos.to_numpy().sum())
    por_grupo = pd.DataFrame(
        {
            "grupo": conteos.index,
            "n_estrellas": conteos.sum(axis=1).to_numpy(),
            "subtipo_mayoritario": conteos.idxmax(axis=1).to_numpy(),
            "pureza_grupo": (conteos.max(axis=1) / conteos.sum(axis=1)).to_numpy(),
        }
    )
    return purity, por_grupo, conteos


def alinear_labels_referencia(labels, subtipos):
    labels = np.asarray(labels)
    subtipos = pd.Series(subtipos).astype(str).to_numpy()
    grupos = np.unique(labels)
    clases = np.unique(subtipos[pd.notna(subtipos)])
    conteos = pd.crosstab(labels, subtipos).reindex(index=grupos, columns=clases, fill_value=0)
    costos = -conteos.to_numpy()
    fila, col = linear_sum_assignment(costos)
    mapping = {grupos[i]: clases[j] for i, j in zip(fila, col)}
    return mapping


def calcular_metricas_clustering(X, labels, sample_size=SAMPLE_SIZE_SILHOUETTE, random_state=RANDOM_STATE_METRICAS):
    labels = np.asarray(labels)
    grupos, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    if len(grupos) < 2:
        return np.nan, np.nan, "grupos_degenerados"
    if len(grupos) >= n:
        return np.nan, np.nan, "un_grupo_por_observacion"

    db = float(davies_bouldin_score(X, labels))
    usado_sample = sample_size is not None and n > sample_size
    silhouette = float(
        silhouette_score(
            X,
            labels,
            sample_size=sample_size if usado_sample else None,
            random_state=random_state,
        )
    )
    estado = f"OK_sample_{sample_size}" if usado_sample else "OK_full"
    return silhouette, db, estado


def auditar_experimento(fila: pd.Series):
    resultado = {
        "experiment_id": fila.get("experiment_id"),
        "modelo": fila.get("modelo"),
        "ruta_experimento": fila.get("ruta_experimento"),
        "artefactos_disponibles": "",
        "puede_calcular_purity": False,
        "puede_calcular_silhouette": False,
        "puede_calcular_davies_bouldin": False,
        "requiere_reentrenamiento": False,
        "motivo": "",
    }
    try:
        carpeta = resolver_ruta(fila["ruta_experimento"])
        config = cargar_config_experimento(carpeta, fila)
        artefactos = {
            "alpha": (carpeta / "modelos" / "alpha.npy").exists(),
            "beta": (carpeta / "modelos" / "beta.npy").exists(),
            "Z": (carpeta / "modelos" / "Z.npy").exists(),
            "labels": (carpeta / "labels.npy").exists(),
            "asignaciones": (carpeta / "asignaciones.npy").exists(),
            "ids": any((carpeta / p).exists() for p in ["modelos/ids.npy", "ids.npy", "modelos/star_ids.npy", "star_ids.npy"]),
            "X_hat": any((carpeta / p).exists() for p in ["modelos/X_hat.npy", "X_hat.npy"]),
            "config": any((carpeta / p).exists() for p in ["config/config.json", "config.json"]),
            "metricas": any((carpeta / p).exists() for p in ["metricas/metricas.json", "metricas/metrics.json", "metricas.json"]),
        }
        resultado["artefactos_disponibles"] = ",".join(k for k, v in artefactos.items() if v)

        labels, extra = obtener_labels_evaluacion(fila["modelo"], carpeta)
        ids_guardados, _ = cargar_ids_guardados(carpeta)
        X, ids = cargar_X_experimento(config, ids_guardados)
        if len(labels) != len(ids) or len(ids) != X.shape[0]:
            raise ValueError(f"Dimensiones inconsistentes labels={len(labels)}, ids={len(ids)}, X={X.shape[0]}")
        if len(set(ids)) != len(ids):
            raise ValueError("IDs duplicados")

        if str(config.get("dataset")).lower() == "rrlyrae":
            subtipos = cargar_subtipos(ids)
            if pd.isna(subtipos).any():
                raise ValueError("Hay subtipos faltantes")
            resultado["puede_calcular_purity"] = True
        resultado["puede_calcular_silhouette"] = True
        resultado["puede_calcular_davies_bouldin"] = True
        resultado["motivo"] = "OK"
    except Exception as exc:
        resultado["requiere_reentrenamiento"] = False
        resultado["motivo"] = str(exc)
    return resultado


def recalcular_experimento(fila: pd.Series):
    carpeta = resolver_ruta(fila["ruta_experimento"])
    config = cargar_config_experimento(carpeta, fila)
    labels, _ = obtener_labels_evaluacion(fila["modelo"], carpeta)
    ids_guardados, _ = cargar_ids_guardados(carpeta)
    X, ids = cargar_X_experimento(config, ids_guardados)
    if len(labels) != len(ids) or X.shape[0] != len(ids):
        raise ValueError("Dimensiones inconsistentes")

    subtipos = cargar_subtipos(ids) if str(config.get("dataset")).lower() == "rrlyrae" else None
    purity, pureza_grupo, conteos = calcular_pureza(labels, subtipos)
    silhouette, db, estado = calcular_metricas_clustering(X, labels)
    return {
        "purity": purity,
        "silhouette": silhouette,
        "davies_bouldin": db,
        "estado_metricas": estado,
        "pureza_por_grupo": pureza_grupo,
        "conteos": conteos,
    }


def crear_backups():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rutas = {}
    for ext in ("csv", "parquet"):
        origen = RESULTS_DIR / f"resumen_experimentos.{ext}"
        if origen.exists():
            destino = RESULTS_DIR / f"resumen_experimentos_backup_{timestamp}.{ext}"
            shutil.copy2(origen, destino)
            rutas[ext] = destino
    return rutas


def actualizar_resumen_metricas(df: pd.DataFrame, resultados: dict):
    df = df.copy()
    if "estado_metricas" not in df.columns:
        df["estado_metricas"] = pd.NA
    for idx, metricas in resultados.items():
        for col in ("purity", "silhouette", "davies_bouldin"):
            actual = df.at[idx, col]
            nuevo = metricas[col]
            if pd.notna(actual) and not np.isclose(float(actual), float(nuevo), equal_nan=True):
                raise ValueError(f"No se sobrescribe {col} en fila {idx}: {actual} != {nuevo}")
            df.at[idx, col] = nuevo
        df.at[idx, "estado_metricas"] = metricas["estado_metricas"]
    return df


def guardar_grafico_purity_db(df: pd.DataFrame):
    salida_dir = RESULTS_DIR / "analisis"
    salida_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "experiment_id",
        "modelo",
        "dataset",
        "discretizacion",
        "n_puntos",
        "normalizacion",
        "inicializacion",
        "optimizacion",
        "K",
        "purity",
        "davies_bouldin",
        "silhouette",
        "n_muestras",
        "estado_metricas",
        "experimento_final",
    ]
    plot_df = df[cols].copy()
    plot_df = plot_df.dropna(subset=["purity", "davies_bouldin", "silhouette"])
    plot_df = plot_df[plot_df["estado_metricas"].astype(str).str.startswith("OK")]
    plot_df.to_csv(salida_dir / "purity_vs_davies_bouldin.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for etiqueta, grupo in plot_df.groupby(plot_df["modelo"].astype(str) + " / " + plot_df["optimizacion"].astype(str)):
        ax.scatter(grupo["davies_bouldin"], grupo["purity"], label=etiqueta, alpha=0.75, s=45)

    seleccion = plot_df[(plot_df["modelo"] == "AA") & (plot_df["experimento_final"] == True)] if "experimento_final" in plot_df else pd.DataFrame()
    if not seleccion.empty:
        fila = seleccion.iloc[-1]
        ax.annotate(
            "AA final",
            (fila["davies_bouldin"], fila["purity"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlabel("Davies-Bouldin (menor es mejor)")
    ax.set_ylabel("Purity (mayor es mejor)")
    ax.set_title("Purity vs Davies-Bouldin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(salida_dir / "purity_vs_davies_bouldin.png", dpi=150)
    plt.close(fig)
    return salida_dir / "purity_vs_davies_bouldin.csv", salida_dir / "purity_vs_davies_bouldin.png"


def ejecutar_recalculo(actualizar=True):
    resumen_csv = RESULTS_DIR / "resumen_experimentos.csv"
    df = pd.read_csv(resumen_csv)
    auditoria = []
    resultados = {}
    for idx, fila in df.iterrows():
        audit = auditar_experimento(fila)
        auditoria.append(audit)
        if (
            audit["puede_calcular_purity"]
            and audit["puede_calcular_silhouette"]
            and audit["puede_calcular_davies_bouldin"]
        ):
            resultados[idx] = recalcular_experimento(fila)

    audit_df = pd.DataFrame(auditoria)
    (RESULTS_DIR / "analisis").mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(RESULTS_DIR / "analisis" / "auditoria_metricas_experimentos.csv", index=False)

    backups = {}
    if actualizar:
        backups = crear_backups()
        df_actualizado = actualizar_resumen_metricas(df, resultados)
        df_actualizado.to_csv(RESULTS_DIR / "resumen_experimentos.csv", index=False)
        df_actualizado.to_parquet(RESULTS_DIR / "resumen_experimentos.parquet", index=False)
        grafico_csv, grafico_png = guardar_grafico_purity_db(df_actualizado)
    else:
        df_actualizado = df
        grafico_csv = grafico_png = None

    return {
        "auditoria": audit_df,
        "resultados": resultados,
        "backups": backups,
        "grafico_csv": grafico_csv,
        "grafico_png": grafico_png,
        "resumen": df_actualizado,
    }


if __name__ == "__main__":
    salida = ejecutar_recalculo(actualizar=True)
    audit = salida["auditoria"]
    print("Auditados:", len(audit))
    print("Recalculables:", int((audit["motivo"] == "OK").sum()))
    print("No recalculables:", int((audit["motivo"] != "OK").sum()))
    print("Backups:", {k: str(v) for k, v in salida["backups"].items()})
    print("Grafico CSV:", salida["grafico_csv"])
    print("Grafico PNG:", salida["grafico_png"])
