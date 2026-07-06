"""
validacion_astronomica.py

Analisis posterior al entrenamiento para cruzar activaciones AA con subtipos
astronomicos RR Lyrae.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR, DATA_DIR


CONFIG_MEJOR_MODELO = {
    "dataset": "rrlyrae",
    "discretizacion": "percentiles",
    "n_puntos": 50,
    "normalizacion": "minmax",
    "suavizado": "none",
    "inicializacion": "random",
    "optimizacion": "nnls",
    "K": 6,
    "n_muestras": 1000,
    "seed": 42,
}

SALIDA_DIR = RESULTS_DIR / "validacion_astronomica"
CATALOGO_SUBTIPOS = RESULTS_DIR / "catalogos" / "rrlyrae_subtipos.csv"


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


def _leer_config_experimento(carpeta_exp: Path) -> dict:
    ruta_config = carpeta_exp / "config" / "config.json"
    with ruta_config.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def _star_id_base(star_id):
    texto = str(star_id)
    for prefijo in ("OGLEIII_", "OGLEIV_"):
        if texto.startswith(prefijo):
            return texto[len(prefijo) :]
    return texto


def _cargar_ids_guardados(carpeta_exp: Path):
    posibles = [
        carpeta_exp / "modelos" / "ids.npy",
        carpeta_exp / "ids.npy",
        carpeta_exp / "modelos" / "star_ids.npy",
        carpeta_exp / "star_ids.npy",
    ]

    for ruta in posibles:
        if ruta.exists():
            return np.load(ruta, allow_pickle=True).astype(str)

    return None


def _reconstruir_ids(config_exp: dict):
    try:
        from config import DATASET, DISCRETIZACION, N_PUNTOS
        from utils.loaders import cargar_dataset

        config_actual_coincide = (
            DATASET == config_exp["dataset"]
            and DISCRETIZACION == config_exp["discretizacion"]
            and N_PUNTOS == config_exp["n_puntos"]
        )
        if config_actual_coincide:
            _, ids = cargar_dataset(
                n_muestras=config_exp.get("n_muestras"),
                seed=int(config_exp.get("seed", 42)),
            )
            return np.asarray(ids, dtype=str)
    except Exception:
        pass

    ruta_datos = (
        DATA_DIR
        / config_exp["discretizacion"]
        / f"{config_exp['dataset']}_{config_exp['n_puntos']}.parquet"
    )
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No existe el dataset discretizado: {ruta_datos}")

    df = pd.read_parquet(ruta_datos, columns=["star_id"])
    ids = df["star_id"].astype(str).to_numpy()
    n_muestras = config_exp.get("n_muestras")

    if n_muestras is None:
        return ids

    n_muestras = int(n_muestras)
    if n_muestras > len(ids):
        raise ValueError(f"n_muestras ({n_muestras}) > total ({len(ids)})")

    rng = np.random.default_rng(int(config_exp.get("seed", 42)))
    idx = rng.choice(len(ids), size=n_muestras, replace=False)
    return ids[idx]


def _filtrar_mejor_modelo(resumen: pd.DataFrame) -> pd.DataFrame:
    filtro = pd.Series(True, index=resumen.index)

    for columna, valor in CONFIG_MEJOR_MODELO.items():
        if columna == "suavizado":
            serie = resumen[columna].fillna("none") if columna in resumen else "none"
            filtro &= serie == valor
        else:
            filtro &= resumen[columna] == valor

    return resumen[filtro].copy()


def cargar_mejor_experimento_rrlyrae():
    ruta_resumen = RESULTS_DIR / "resumen_experimentos.csv"
    if not ruta_resumen.exists():
        raise FileNotFoundError(f"No existe {ruta_resumen}")

    resumen = pd.read_csv(ruta_resumen)
    candidatos = _filtrar_mejor_modelo(resumen)
    if candidatos.empty:
        raise ValueError("No se encontro un experimento con la configuracion candidata.")

    if "fecha" in candidatos:
        candidatos = candidatos.sort_values("fecha")

    fila = candidatos.iloc[-1]
    carpeta_exp = _resolver_experimento(fila["ruta_experimento"])
    alpha = np.load(carpeta_exp / "modelos" / "alpha.npy")

    ids = _cargar_ids_guardados(carpeta_exp)
    config_exp = _leer_config_experimento(carpeta_exp)
    if ids is None:
        ids = _reconstruir_ids(config_exp)

    if len(ids) != alpha.shape[0]:
        raise ValueError(
            f"Cantidad de ids ({len(ids)}) no coincide con alpha ({alpha.shape[0]})."
        )

    return {
        "ruta_experimento": carpeta_exp,
        "config": config_exp,
        "alpha": alpha,
        "ids": ids,
        "fila_resumen": fila.to_dict(),
    }


def construir_tabla_alpha_subtipo(ruta_experimento):
    carpeta_exp = _resolver_experimento(ruta_experimento)
    config_exp = _leer_config_experimento(carpeta_exp)
    alpha = np.load(carpeta_exp / "modelos" / "alpha.npy")
    ids = _cargar_ids_guardados(carpeta_exp)
    if ids is None:
        ids = _reconstruir_ids(config_exp)

    if len(ids) != alpha.shape[0]:
        raise ValueError(
            f"Cantidad de ids ({len(ids)}) no coincide con alpha ({alpha.shape[0]})."
        )

    K = alpha.shape[1]
    tabla = pd.DataFrame({"star_id": ids.astype(str)})
    for k in range(K):
        tabla[f"alpha_{k + 1}"] = alpha[:, k]

    tabla["arquetipo_dominante"] = np.argmax(alpha, axis=1) + 1
    tabla["star_id_base"] = tabla["star_id"].map(_star_id_base)

    catalogo = pd.read_csv(CATALOGO_SUBTIPOS)
    columnas_catalogo = [
        "star_id",
        "star_id_base",
        "subtipo_rrlyrae",
        "archivo_origen",
    ]
    catalogo = catalogo[columnas_catalogo].drop_duplicates()

    tabla = tabla.merge(
        catalogo.drop(columns=["star_id_base"]),
        on="star_id",
        how="left",
    )

    faltantes = tabla["subtipo_rrlyrae"].isna()
    if faltantes.any():
        catalogo_base = (
            catalogo[["star_id_base", "subtipo_rrlyrae", "archivo_origen"]]
            .drop_duplicates("star_id_base")
        )
        relleno = tabla.loc[faltantes, ["star_id_base"]].merge(
            catalogo_base,
            on="star_id_base",
            how="left",
        )
        tabla.loc[faltantes, "subtipo_rrlyrae"] = relleno["subtipo_rrlyrae"].to_numpy()
        tabla.loc[faltantes, "archivo_origen"] = relleno["archivo_origen"].to_numpy()

    columnas = (
        ["star_id", "subtipo_rrlyrae"]
        + [f"alpha_{k + 1}" for k in range(K)]
        + ["arquetipo_dominante", "star_id_base", "archivo_origen"]
    )
    return tabla[columnas]


def calcular_validacion_astronomica(ruta_experimento=None, guardar=True, salida_dir=None):
    if ruta_experimento is None:
        mejor = cargar_mejor_experimento_rrlyrae()
        ruta_experimento = mejor["ruta_experimento"]

    tabla = construir_tabla_alpha_subtipo(ruta_experimento)
    alpha_cols = [c for c in tabla.columns if c.startswith("alpha_")]
    con_subtipo = tabla.dropna(subset=["subtipo_rrlyrae"]).copy()

    activacion_media = (
        con_subtipo.groupby("subtipo_rrlyrae")[alpha_cols].mean().reset_index()
    )

    conteos = pd.crosstab(
        con_subtipo["subtipo_rrlyrae"],
        con_subtipo["arquetipo_dominante"],
    )
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0).fillna(0)
    dominante_por_subtipo = porcentajes.reset_index()

    distribucion = pd.crosstab(
        con_subtipo["arquetipo_dominante"],
        con_subtipo["subtipo_rrlyrae"],
    )
    distribucion_pct = distribucion.div(distribucion.sum(axis=1), axis=0).fillna(0)

    pureza = pd.DataFrame(
        {
            "arquetipo_dominante": distribucion.index,
            "n_estrellas": distribucion.sum(axis=1).to_numpy(),
            "subtipo_mayoritario": distribucion.idxmax(axis=1).to_numpy(),
            "pureza": distribucion_pct.max(axis=1).to_numpy(),
        }
    )
    for subtipo in distribucion_pct.columns:
        pureza[f"pct_{subtipo}"] = distribucion_pct[subtipo].to_numpy()
        pureza[f"n_{subtipo}"] = distribucion[subtipo].to_numpy()

    resultados = {
        "alpha_subtipos": tabla,
        "activacion_media_por_subtipo": activacion_media,
        "arquetipo_dominante_por_subtipo": dominante_por_subtipo,
        "pureza_por_arquetipo": pureza,
        "matriz_subtipo_arquetipo_conteos": conteos,
        "matriz_subtipo_arquetipo_porcentajes": porcentajes,
    }

    if guardar:
        guardar_resultados(resultados, salida_dir=salida_dir)
        guardar_figuras(resultados, salida_dir=salida_dir)

    return resultados


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


def guardar_resultados(resultados, salida_dir=None):
    salida_dir = Path(salida_dir) if salida_dir is not None else SALIDA_DIR
    salida_dir.mkdir(parents=True, exist_ok=True)
    for nombre, df in resultados.items():
        guardar_indice = nombre.startswith("matriz_")
        df.to_csv(salida_dir / f"{nombre}.csv", index=guardar_indice)


def guardar_figuras(resultados, salida_dir=None):
    salida_dir = Path(salida_dir) if salida_dir is not None else SALIDA_DIR
    salida_dir.mkdir(parents=True, exist_ok=True)

    activacion = resultados["activacion_media_por_subtipo"].set_index("subtipo_rrlyrae")
    _plot_heatmap(
        activacion,
        salida_dir / "heatmap_activacion_media_por_subtipo.png",
        "Activacion media por subtipo",
    )

    dominante = resultados["arquetipo_dominante_por_subtipo"].set_index(
        "subtipo_rrlyrae"
    )
    _plot_heatmap(
        dominante,
        salida_dir / "heatmap_arquetipo_dominante_por_subtipo.png",
        "Arquetipo dominante por subtipo",
    )

    pureza_cols = [c for c in resultados["pureza_por_arquetipo"].columns if c.startswith("pct_")]
    pureza = resultados["pureza_por_arquetipo"].set_index("arquetipo_dominante")[
        pureza_cols
    ]
    pureza.columns = [c.replace("pct_", "") for c in pureza.columns]
    _plot_heatmap(
        pureza,
        salida_dir / "heatmap_pureza_por_arquetipo.png",
        "Distribucion de subtipos por arquetipo dominante",
    )


def main():
    resultados = calcular_validacion_astronomica()
    tabla = resultados["alpha_subtipos"]
    n_cruzadas = int(tabla["subtipo_rrlyrae"].notna().sum())
    n_sin_subtipo = int(tabla["subtipo_rrlyrae"].isna().sum())

    print(f"Directorio salida: {SALIDA_DIR}")
    print(f"Estrellas cruzadas con subtipo: {n_cruzadas}")
    print(f"Estrellas sin subtipo: {n_sin_subtipo}")
    print("\nSubtipos encontrados:")
    print(tabla["subtipo_rrlyrae"].value_counts(dropna=False).to_string())
    print("\nActivacion media por subtipo:")
    print(resultados["activacion_media_por_subtipo"].to_string(index=False))
    print("\nPureza por arquetipo:")
    print(resultados["pureza_por_arquetipo"].to_string(index=False))


if __name__ == "__main__":
    main()
