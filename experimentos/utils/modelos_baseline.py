"""
modelos_baseline.py

Baselines comparables con la infraestructura experimental.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from utils.exportacion import actualizar_resumen_experimentos, config_serializable
from utils.optimizacion import _obtener_nnls, project_to_simplex


def ejecutar_kmeans(X, K, seed=42):
    inicio = time.perf_counter()
    modelo = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    labels = modelo.fit_predict(X)
    centroides = modelo.cluster_centers_
    X_hat = centroides[labels]
    tiempo_segundos = time.perf_counter() - inicio

    error_fro = float(np.linalg.norm(X - X_hat, ord="fro"))
    error_rel = float(np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-8))
    error_per_sample = np.linalg.norm(X - X_hat, axis=1)

    return {
        "modelo": modelo,
        "labels": labels,
        "centroides": centroides,
        "X_hat": X_hat,
        "error_fro": error_fro,
        "error_rel": error_rel,
        "error_per_sample": error_per_sample,
        "tiempo_segundos": tiempo_segundos,
    }


def _fase_duplicada(m):
    fase = np.linspace(0.0, 1.0, m)
    return np.concatenate([fase, fase + 1.0])


def _curva_duplicada(curva):
    curva = np.asarray(curva)
    return np.concatenate([curva, curva])


def plot_centroides_duplicados(centroides, carpeta_figuras):
    carpeta = Path(carpeta_figuras) / "arquetipos"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "centroides_duplicados.png"

    fase = _fase_duplicada(centroides.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, centroide in enumerate(centroides):
        ax.plot(fase, _curva_duplicada(centroide), label=f"Cluster {k + 1}")

    ax.set_xlabel("Fase")
    ax.set_ylabel("Magnitud normalizada")
    ax.set_title("Centroides K-Means duplicados")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def plot_aproximaciones_baseline(X, X_hat, ids, carpeta_figuras, indices=None):
    carpeta = Path(carpeta_figuras) / "reconstrucciones"
    carpeta.mkdir(parents=True, exist_ok=True)
    indices = [0, 1, 2] if indices is None else indices
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
            label="Aproximacion",
            linewidth=2,
            linestyle="--",
        )
        ax.set_xlabel("Fase")
        ax.set_ylabel("Magnitud normalizada")
        ax.set_title(f"Aproximacion {indice:03d} | {ids[indice]}")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        rutas.append(ruta)

    return rutas


def plot_reconstrucciones_kmeans(X, X_hat, ids, carpeta_figuras, indices=None):
    return plot_aproximaciones_baseline(X, X_hat, ids, carpeta_figuras, indices=indices)


def plot_pca_labels(X, labels, carpeta_figuras):
    carpeta = Path(carpeta_figuras) / "proyecciones"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "pca_labels.png"

    coords = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.6, cmap="tab10")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA de curvas coloreada por cluster")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def plot_representantes_duplicados(representantes, carpeta_figuras, nombre_archivo, titulo):
    carpeta = Path(carpeta_figuras) / "arquetipos"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre_archivo

    fase = _fase_duplicada(representantes.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, representante in enumerate(representantes):
        ax.plot(fase, _curva_duplicada(representante), label=f"Representante {k + 1}")

    ax.set_xlabel("Fase")
    ax.set_ylabel("Magnitud normalizada")
    ax.set_title(titulo)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def _seleccionar_indices_mas_cercanos(X, centroides):
    distancias = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)
    seleccionados = []

    for k in range(centroides.shape[0]):
        for candidato in np.argsort(distancias[:, k]):
            candidato = int(candidato)
            if candidato not in seleccionados:
                seleccionados.append(candidato)
                break

    return np.asarray(seleccionados, dtype=int)


def _calcular_alpha_nnls(X, representantes):
    nnls = _obtener_nnls()
    alpha = np.zeros((X.shape[0], representantes.shape[0]))

    for i in range(X.shape[0]):
        pesos, _ = nnls(representantes.T, X[i])
        alpha[i] = project_to_simplex(pesos)

    return alpha


def ejecutar_ada(X, K, seed=42):
    """
    Baseline ADA (Archetypoid Analysis).

    Referencia: Vinue, G., Epifanio, I. y Alemany, S. (2015), Archetypoids:
    A new approach to define representative archetypal data. Computational
    Statistics & Data Analysis.

    En ADA los representantes deben ser observaciones reales del dataset. Esta
    implementacion usa centroides K-Means como guia reproducible para escoger
    observaciones reales cercanas y reconstruye cada curva como combinacion
    convexa de esos arquetipoides mediante NNLS proyectado al simplex.
    """
    inicio = time.perf_counter()
    guia = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    guia.fit(X)

    indices = _seleccionar_indices_mas_cercanos(X, guia.cluster_centers_)
    arquetipoides = X[indices]
    alpha = _calcular_alpha_nnls(X, arquetipoides)
    asignaciones = np.argmax(alpha, axis=1)
    X_hat = alpha @ arquetipoides
    tiempo_segundos = time.perf_counter() - inicio

    error_fro = float(np.linalg.norm(X - X_hat, ord="fro"))
    error_rel = float(np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-8))
    error_per_sample = np.linalg.norm(X - X_hat, axis=1)

    return {
        "modelo_guia": guia,
        "indices": indices,
        "arquetipoides": arquetipoides,
        "alpha": alpha,
        "asignaciones": asignaciones,
        "X_hat": X_hat,
        "error_fro": error_fro,
        "error_rel": error_rel,
        "error_per_sample": error_per_sample,
        "tiempo_segundos": tiempo_segundos,
    }


def _star_id_base(star_id):
    texto = str(star_id)
    for prefijo in ("OGLEIII_", "OGLEIV_"):
        if texto.startswith(prefijo):
            return texto[len(prefijo) :]
    return texto


def _plot_heatmap(df, ruta, titulo, formato=".2f"):
    data = df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(df.index) + 2)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    ax.set_title(titulo)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], formato), ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def validar_kmeans_subtipos(ids, labels, carpeta_salida):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    catalogo = pd.read_csv(RESULTS_DIR / "catalogos" / "rrlyrae_subtipos.csv")
    catalogo = catalogo[["star_id", "star_id_base", "subtipo_rrlyrae"]].drop_duplicates()

    tabla = pd.DataFrame(
        {
            "star_id": np.asarray(ids, dtype=str),
            "cluster": labels.astype(int) + 1,
        }
    )
    tabla["star_id_base"] = tabla["star_id"].map(_star_id_base)
    tabla = tabla.merge(
        catalogo.drop(columns=["star_id_base"]),
        on="star_id",
        how="left",
    )

    faltantes = tabla["subtipo_rrlyrae"].isna()
    if faltantes.any():
        catalogo_base = catalogo[["star_id_base", "subtipo_rrlyrae"]].drop_duplicates(
            "star_id_base"
        )
        relleno = tabla.loc[faltantes, ["star_id_base"]].merge(
            catalogo_base,
            on="star_id_base",
            how="left",
        )
        tabla.loc[faltantes, "subtipo_rrlyrae"] = relleno["subtipo_rrlyrae"].to_numpy()

    con_subtipo = tabla.dropna(subset=["subtipo_rrlyrae"]).copy()
    conteos = pd.crosstab(con_subtipo["subtipo_rrlyrae"], con_subtipo["cluster"])
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0).fillna(0)
    cluster_por_subtipo = porcentajes.reset_index()

    distribucion = pd.crosstab(con_subtipo["cluster"], con_subtipo["subtipo_rrlyrae"])
    distribucion_pct = distribucion.div(distribucion.sum(axis=1), axis=0).fillna(0)
    pureza = pd.DataFrame(
        {
            "cluster": distribucion.index,
            "n_estrellas": distribucion.sum(axis=1).to_numpy(),
            "subtipo_mayoritario": distribucion.idxmax(axis=1).to_numpy(),
            "pureza": distribucion_pct.max(axis=1).to_numpy(),
        }
    )
    for subtipo in distribucion_pct.columns:
        pureza[f"pct_{subtipo}"] = distribucion_pct[subtipo].to_numpy()
        pureza[f"n_{subtipo}"] = distribucion[subtipo].to_numpy()

    tabla.to_csv(carpeta_salida / "labels_subtipos.csv", index=False)
    cluster_por_subtipo.to_csv(
        carpeta_salida / "cluster_dominante_por_subtipo.csv",
        index=False,
    )
    pureza.to_csv(carpeta_salida / "pureza_por_cluster.csv", index=False)
    conteos.to_csv(carpeta_salida / "matriz_subtipo_cluster_conteos.csv")
    porcentajes.to_csv(carpeta_salida / "matriz_subtipo_cluster_porcentajes.csv")

    _plot_heatmap(
        porcentajes,
        carpeta_salida / "heatmap_cluster_dominante_por_subtipo.png",
        "Cluster por subtipo",
    )
    pureza_cols = [c for c in pureza.columns if c.startswith("pct_")]
    pureza_heatmap = pureza.set_index("cluster")[pureza_cols]
    pureza_heatmap.columns = [c.replace("pct_", "") for c in pureza_heatmap.columns]
    _plot_heatmap(
        pureza_heatmap,
        carpeta_salida / "heatmap_pureza_por_cluster.png",
        "Distribucion de subtipos por cluster",
    )

    return {
        "labels_subtipos": tabla,
        "cluster_dominante_por_subtipo": cluster_por_subtipo,
        "pureza_por_cluster": pureza,
        "matriz_subtipo_cluster_conteos": conteos,
        "matriz_subtipo_cluster_porcentajes": porcentajes,
    }


def validar_ada_subtipos(ids, asignaciones, carpeta_salida):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    catalogo = pd.read_csv(RESULTS_DIR / "catalogos" / "rrlyrae_subtipos.csv")
    catalogo = catalogo[["star_id", "star_id_base", "subtipo_rrlyrae"]].drop_duplicates()

    tabla = pd.DataFrame(
        {
            "star_id": np.asarray(ids, dtype=str),
            "arquetipoide": asignaciones.astype(int) + 1,
        }
    )
    tabla["star_id_base"] = tabla["star_id"].map(_star_id_base)
    tabla = tabla.merge(
        catalogo.drop(columns=["star_id_base"]),
        on="star_id",
        how="left",
    )

    faltantes = tabla["subtipo_rrlyrae"].isna()
    if faltantes.any():
        catalogo_base = catalogo[["star_id_base", "subtipo_rrlyrae"]].drop_duplicates(
            "star_id_base"
        )
        relleno = tabla.loc[faltantes, ["star_id_base"]].merge(
            catalogo_base,
            on="star_id_base",
            how="left",
        )
        tabla.loc[faltantes, "subtipo_rrlyrae"] = relleno["subtipo_rrlyrae"].to_numpy()

    con_subtipo = tabla.dropna(subset=["subtipo_rrlyrae"]).copy()
    conteos = pd.crosstab(con_subtipo["subtipo_rrlyrae"], con_subtipo["arquetipoide"])
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0).fillna(0)
    arquetipoide_por_subtipo = porcentajes.reset_index()

    distribucion = pd.crosstab(
        con_subtipo["arquetipoide"],
        con_subtipo["subtipo_rrlyrae"],
    )
    distribucion_pct = distribucion.div(distribucion.sum(axis=1), axis=0).fillna(0)
    pureza = pd.DataFrame(
        {
            "arquetipoide": distribucion.index,
            "n_estrellas": distribucion.sum(axis=1).to_numpy(),
            "subtipo_mayoritario": distribucion.idxmax(axis=1).to_numpy(),
            "pureza": distribucion_pct.max(axis=1).to_numpy(),
        }
    )
    for subtipo in distribucion_pct.columns:
        pureza[f"pct_{subtipo}"] = distribucion_pct[subtipo].to_numpy()
        pureza[f"n_{subtipo}"] = distribucion[subtipo].to_numpy()

    tabla.to_csv(carpeta_salida / "asignaciones_subtipos.csv", index=False)
    arquetipoide_por_subtipo.to_csv(
        carpeta_salida / "arquetipoide_dominante_por_subtipo.csv",
        index=False,
    )
    pureza.to_csv(carpeta_salida / "pureza_por_arquetipoide.csv", index=False)
    conteos.to_csv(carpeta_salida / "matriz_subtipo_arquetipoide_conteos.csv")
    porcentajes.to_csv(carpeta_salida / "matriz_subtipo_arquetipoide_porcentajes.csv")

    _plot_heatmap(
        porcentajes,
        carpeta_salida / "heatmap_arquetipoide_dominante_por_subtipo.png",
        "Arquetipoide por subtipo",
    )
    pureza_cols = [c for c in pureza.columns if c.startswith("pct_")]
    pureza_heatmap = pureza.set_index("arquetipoide")[pureza_cols]
    pureza_heatmap.columns = [c.replace("pct_", "") for c in pureza_heatmap.columns]
    _plot_heatmap(
        pureza_heatmap,
        carpeta_salida / "heatmap_pureza_por_arquetipoide.png",
        "Distribucion de subtipos por arquetipoide",
    )

    return {
        "asignaciones_subtipos": tabla,
        "arquetipoide_dominante_por_subtipo": arquetipoide_por_subtipo,
        "pureza_por_arquetipoide": pureza,
        "matriz_subtipo_arquetipoide_conteos": conteos,
        "matriz_subtipo_arquetipoide_porcentajes": porcentajes,
    }


def cargar_rrlyrae_final():
    ruta = DATA_DIR / "percentiles" / "rrlyrae_50.parquet"
    df = pd.read_parquet(ruta)
    ids = df["star_id"].astype(str).to_numpy()
    X = df.drop(columns=["star_id"]).to_numpy()
    X_min = np.min(X, axis=1, keepdims=True)
    X_max = np.max(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    return X, ids


def ejecutar_baseline_kmeans_rrlyrae_final():
    config = {
        "modelo": "KMEANS",
        "dataset": "rrlyrae",
        "discretizacion": "percentiles",
        "n_puntos": 50,
        "n_muestras": None,
        "normalizacion": "minmax",
        "suavizado": "none",
        "inicializacion": "kmeans++",
        "optimizacion": "kmeans",
        "K": 6,
        "max_iter": 300,
        "seed": 42,
    }
    carpeta = RESULTS_DIR / "baselines" / "kmeans"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "arquetipos").mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "reconstrucciones").mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "proyecciones").mkdir(parents=True, exist_ok=True)

    X, ids = cargar_rrlyrae_final()
    resultado = ejecutar_kmeans(X, K=config["K"], seed=config["seed"])
    labels = resultado["labels"]
    centroides = resultado["centroides"]
    X_hat = resultado["X_hat"]

    metricas = {
        "error_fro": resultado["error_fro"],
        "error_rel": resultado["error_rel"],
        "error_per_sample": resultado["error_per_sample"],
        "tiempo_segundos": resultado["tiempo_segundos"],
    }

    with open(carpeta / "config.json", "w", encoding="utf-8") as archivo:
        json.dump(config, archivo, indent=4, ensure_ascii=False)
    with open(carpeta / "metricas.json", "w", encoding="utf-8") as archivo:
        json.dump(config_serializable(metricas), archivo, indent=4, ensure_ascii=False)

    np.save(carpeta / "labels.npy", labels)
    np.save(carpeta / "centroides.npy", centroides)
    np.save(carpeta / "X_hat.npy", X_hat)
    np.save(carpeta / "ids.npy", ids)

    figuras = {
        "centroides": plot_centroides_duplicados(centroides, carpeta / "figuras"),
        "reconstrucciones": plot_reconstrucciones_kmeans(X, X_hat, ids, carpeta / "figuras"),
        "pca_labels": plot_pca_labels(X, labels, carpeta / "figuras"),
    }

    resumen = actualizar_resumen_experimentos(
        config=config,
        metrics=metricas,
        carpeta_exp=carpeta,
        tiempo_segundos=resultado["tiempo_segundos"],
        X=X,
        metadata={
            "estado": "OK",
            "convergio": True,
            "iteraciones_reales": int(resultado["modelo"].n_iter_),
            "tiempo_por_iteracion": resultado["tiempo_segundos"] / resultado["modelo"].n_iter_,
            "version_algoritmo": "KMeans_sklearn",
            "loss_final": resultado["error_fro"],
            "experimento_final": False,
        },
    )

    validacion = validar_kmeans_subtipos(ids, labels, carpeta / "validacion_astronomica")

    return {
        "config": config,
        "carpeta": carpeta,
        "n_estrellas": int(X.shape[0]),
        "metricas": metricas,
        "labels": labels,
        "centroides": centroides,
        "figuras": figuras,
        "resumen": resumen,
        "validacion_astronomica": validacion,
        "tiempo_segundos": resultado["tiempo_segundos"],
    }


def ejecutar_baseline_ada_rrlyrae_final():
    config = {
        "modelo": "ADA",
        "dataset": "rrlyrae",
        "discretizacion": "percentiles",
        "n_puntos": 50,
        "n_muestras": None,
        "normalizacion": "minmax",
        "suavizado": "none",
        "inicializacion": "kmeans++_nearest_observation",
        "optimizacion": "ada_nnls_simplex",
        "K": 6,
        "max_iter": None,
        "seed": 42,
    }
    carpeta = RESULTS_DIR / "baselines" / "ada"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "arquetipos").mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "reconstrucciones").mkdir(parents=True, exist_ok=True)
    (carpeta / "figuras" / "proyecciones").mkdir(parents=True, exist_ok=True)

    X, ids = cargar_rrlyrae_final()
    resultado = ejecutar_ada(X, K=config["K"], seed=config["seed"])
    asignaciones = resultado["asignaciones"]
    arquetipoides = resultado["arquetipoides"]
    X_hat = resultado["X_hat"]

    metricas = {
        "error_fro": resultado["error_fro"],
        "error_rel": resultado["error_rel"],
        "error_per_sample": resultado["error_per_sample"],
        "tiempo_segundos": resultado["tiempo_segundos"],
    }

    with open(carpeta / "config.json", "w", encoding="utf-8") as archivo:
        json.dump(config, archivo, indent=4, ensure_ascii=False)
    with open(carpeta / "metricas.json", "w", encoding="utf-8") as archivo:
        json.dump(config_serializable(metricas), archivo, indent=4, ensure_ascii=False)

    np.save(carpeta / "asignaciones.npy", asignaciones)
    np.save(carpeta / "alpha.npy", resultado["alpha"])
    np.save(carpeta / "arquetipoides.npy", arquetipoides)
    np.save(carpeta / "arquetipoide_indices.npy", resultado["indices"])
    np.save(carpeta / "arquetipoide_ids.npy", ids[resultado["indices"]])
    np.save(carpeta / "X_hat.npy", X_hat)
    np.save(carpeta / "ids.npy", ids)

    figuras = {
        "arquetipoides": plot_representantes_duplicados(
            arquetipoides,
            carpeta / "figuras",
            "arquetipoides_duplicados.png",
            "Arquetipoides ADA duplicados",
        ),
        "reconstrucciones": plot_reconstrucciones_kmeans(X, X_hat, ids, carpeta / "figuras"),
        "pca_asignaciones": plot_pca_labels(X, asignaciones, carpeta / "figuras"),
    }

    resumen = actualizar_resumen_experimentos(
        config=config,
        metrics=metricas,
        carpeta_exp=carpeta,
        tiempo_segundos=resultado["tiempo_segundos"],
        X=X,
        metadata={
            "estado": "OK",
            "convergio": None,
            "iteraciones_reales": None,
            "tiempo_por_iteracion": None,
            "version_algoritmo": "ADA_archetypoids_nnls_simplex",
            "loss_final": resultado["error_fro"],
            "experimento_final": False,
        },
    )

    validacion = validar_ada_subtipos(ids, asignaciones, carpeta / "validacion_astronomica")

    return {
        "config": config,
        "carpeta": carpeta,
        "n_estrellas": int(X.shape[0]),
        "metricas": metricas,
        "asignaciones": asignaciones,
        "arquetipoides": arquetipoides,
        "arquetipoide_indices": resultado["indices"],
        "arquetipoide_ids": ids[resultado["indices"]],
        "figuras": figuras,
        "resumen": resumen,
        "validacion_astronomica": validacion,
        "tiempo_segundos": resultado["tiempo_segundos"],
    }


if __name__ == "__main__":
    resultado = ejecutar_baseline_kmeans_rrlyrae_final()
    print(f"Carpeta: {resultado['carpeta']}")
    print(f"N estrellas: {resultado['n_estrellas']}")
    print(f"error_rel: {resultado['metricas']['error_rel']}")
    print(f"error_fro: {resultado['metricas']['error_fro']}")
    print(f"tiempo_segundos: {resultado['tiempo_segundos']}")
    print("\nPureza por cluster:")
    print(resultado["validacion_astronomica"]["pureza_por_cluster"].to_string(index=False))
