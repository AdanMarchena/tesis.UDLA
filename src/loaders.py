from pathlib import Path
import pandas as pd

# Carpeta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carpeta donde se encuentran los datasets procesados
DATASETS_DIR = PROJECT_ROOT / "data" / "processed" / "discretizaciones"

# Ruta de archivos
PARTICIONES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "particiones"
    / "particiones_principal_60_20_20.parquet"
)

# Archivos
ARCHIVOS_DATASETS = {
    "interpolacion_50": "interpolacion_50.parquet",
    "interpolacion_100": "interpolacion_100.parquet",
    "medianas_fase_50": "medianas_fase_50.parquet",
    "medianas_fase_100": "medianas_fase_100.parquet",
}

# Si el archivo no existe, se lanza una excepción
def cargar_dataset(nombre_dataset):
    """Cargar las magnitudes y los metadatos separados por partición."""

    # Comprobar que el nombre corresponde a un dataset disponible.
    if nombre_dataset not in ARCHIVOS_DATASETS:
        raise ValueError(f"Dataset no reconocido: {nombre_dataset}")

    # Leer el dataset y las particiones previamente guardadas.
    ruta_dataset = DATASETS_DIR / ARCHIVOS_DATASETS[nombre_dataset]
    df_dataset = pd.read_parquet(ruta_dataset)
    df_particiones = pd.read_parquet(PARTICIONES_PATH)

    # Añadir las magnitudes a las curvas del conjunto principal.
    df_completo = df_particiones.merge(
        df_dataset,
        on="observacion_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )

    # Evitar continuar si alguna curva de las particiones no se encontró.
    if not df_completo["_merge"].eq("both").all():
        raise ValueError(
            "Hay curvas del archivo de particiones ausentes en el dataset."
        )

    df_completo = df_completo.drop(columns="_merge")

    # Identificar las columnas de magnitud en su orden de fase.
    n_puntos = int(nombre_dataset.rsplit("_", 1)[1])
    columnas_magnitud = [
        f"magnitud_{i:03d}" for i in range(n_puntos)
    ]
    columnas_metadata = df_particiones.columns.tolist()

    # Comprobar que todas las filas tengan una partición válida.
    if not df_completo["particion"].isin(["train", "val", "test"]).all():
        raise ValueError("El archivo contiene particiones no válidas.")

    # Preparar las magnitudes y los metadatos de cada conjunto.
    datos = {}

    for particion in ("train", "val", "test"):
        filas = df_completo.loc[
            df_completo["particion"] == particion
        ]

        datos[particion] = {
            "X": filas[columnas_magnitud].to_numpy(
                dtype="float32", copy=True
            ),
            "metadata": filas[columnas_metadata]
            .reset_index(drop=True)
            .copy(),
        }

    return datos

if __name__ == "__main__":
    datos = cargar_dataset("interpolacion_50")

    for particion, contenido in datos.items():
        print(
            f"{particion}: "
            f"{contenido['X'].shape[0]:,} curvas, "
            f"{contenido['X'].shape[1]} puntos"
        )
