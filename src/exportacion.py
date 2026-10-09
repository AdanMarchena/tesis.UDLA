import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "resultados"

REGISTRO_CSV = (
    RESULTS_DIR
    / "registro_experimentos.csv"
)

REGISTRO_PARQUET = (
    RESULTS_DIR
    / "registro_experimentos.parquet"
)

EXPERIMENTS_DIR = (
    RESULTS_DIR
    / "experimentos"
)


def generate_experiment_id(configuracion):
    """
    Generar un identificador reproducible
    desde la configuración.
    """

    configuracion_ordenada = json.dumps(
        configuracion,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        configuracion_ordenada.encode("utf-8")
    ).hexdigest()[:16]


def experiment_is_registered(experiment_id):
    """
    Comprobar si un experimento ya está
    en la tabla de resultados.
    """

    if not REGISTRO_CSV.exists():
        return False

    ids_registrados = pd.read_csv(
        REGISTRO_CSV,
        usecols=["experiment_id"],
        dtype={"experiment_id": str},
    )

    return str(experiment_id) in set(
        ids_registrados["experiment_id"]
    )


def save_experiment_record(registro):
    """
    Agregar o actualizar una ejecución
    en la tabla de resultados.
    """

    if "experiment_id" not in registro:
        raise ValueError(
            "El registro debe contener experiment_id."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nueva_fila = pd.DataFrame(
        [registro]
    )

    if REGISTRO_CSV.exists():
        tabla = pd.read_csv(
            REGISTRO_CSV
        )

        if "experiment_id" not in tabla.columns:
            raise ValueError(
                "El registro existente no contiene "
                "experiment_id."
            )

        tabla = tabla.loc[
            tabla["experiment_id"].astype(str)
            != str(registro["experiment_id"])
        ]

        tabla = pd.concat(
            [
                tabla,
                nueva_fila,
            ],
            ignore_index=True,
        )
    else:
        tabla = nueva_fila

    tabla.to_csv(
        REGISTRO_CSV,
        index=False,
    )

    tabla.to_parquet(
        REGISTRO_PARQUET,
        index=False,
    )

    return tabla


def save_experiment_artifacts(
    experiment_id,
    configuracion,
    patrones,
    grupos_train,
    grupos_val,
    ids_train,
    ids_val,
    alphas_train=None,
    alphas_val=None,
    betas=None,
):
    """
    Guardar los artefactos de una ejecución.

    Los coeficientes alpha y beta son
    opcionales porque solo corresponden
    al análisis arquetípico.
    """

    carpeta_experimento = (
        EXPERIMENTS_DIR
        / experiment_id
    )

    carpeta_experimento.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_config = (
        carpeta_experimento
        / "config.json"
    )

    with open(
        ruta_config,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    np.save(
        carpeta_experimento
        / "patrones.npy",
        np.asarray(patrones),
    )

    np.save(
        carpeta_experimento
        / "grupos_train.npy",
        np.asarray(grupos_train),
    )

    np.save(
        carpeta_experimento
        / "grupos_val.npy",
        np.asarray(grupos_val),
    )

    np.save(
        carpeta_experimento
        / "ids_train.npy",
        np.asarray(
            ids_train,
            dtype=str,
        ),
    )

    np.save(
        carpeta_experimento
        / "ids_val.npy",
        np.asarray(
            ids_val,
            dtype=str,
        ),
    )

    if alphas_train is not None:
        np.save(
            carpeta_experimento
            / "alphas_train.npy",
            np.asarray(
                alphas_train,
                dtype=np.float64,
            ),
        )

    if alphas_val is not None:
        np.save(
            carpeta_experimento
            / "alphas_val.npy",
            np.asarray(
                alphas_val,
                dtype=np.float64,
            ),
        )

    if betas is not None:
        np.save(
            carpeta_experimento
            / "betas.npy",
            np.asarray(
                betas,
                dtype=np.float64,
            ),
        )

    return carpeta_experimento

