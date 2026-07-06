"""
loaders.py

Carga datasets discretizados para experimentos de Análisis Arquetípico.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    DATASET,
    DISCRETIZACION,
    N_PUNTOS,
)

def _obtener_ruta():

    carpeta = DATA_DIR / DISCRETIZACION

    archivo = f"{DATASET}_{N_PUNTOS}.parquet"

    ruta = carpeta / archivo

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo:\n{ruta}")

    return ruta

def cargar_dataset(n_muestras=None, seed=42):

    ruta = _obtener_ruta()

    print("=" * 50)
    print(f"Cargando dataset:\n{ruta}")
    print("=" * 50)

    df = pd.read_parquet(ruta)

    ids = df["star_id"].values
    X = df.drop(columns=["star_id"]).values

    n_total = X.shape[0]

    # ==========================================
    # Submuestreo opcional
    # ==========================================
    if n_muestras is not None:

        if n_muestras > n_total:
            raise ValueError(
                f"n_muestras ({n_muestras}) > total ({n_total})"
            )

        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_muestras, replace=False)

        X = X[idx]
        ids = ids[idx]

        print(f"Submuestreo activado: {n_muestras} estrellas")

    # ==========================================
    # Resumen
    # ==========================================
    print(f"Estrellas cargadas : {X.shape[0]:,}")
    print(f"Variables           : {X.shape[1]:,}")
    print("=" * 50)

    return X, ids