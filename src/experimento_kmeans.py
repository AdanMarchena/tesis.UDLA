from datetime import datetime, timezone
from time import perf_counter

import sklearn

from loaders import cargar_dataset
from normalizacion import aplicar_normalizacion
from modelos import kmeans_model, kmeans_reconstruction
from metricas import (
    mse_metric,
    rss_metric,
    calcular_metricas_ejecucion,
)
from exportacion import (
    PROJECT_ROOT,
    generate_experiment_id,
    experiment_is_registered,
    save_experiment_record,
    save_experiment_artifacts,
)


KMEANS_N_INIT = 1
KMEANS_MAX_ITER = 300
KMEANS_TOL = 1e-4
KMEANS_ALGORITHM = "lloyd"

SILHOUETTE_N_MUESTRAS = 5000
SILHOUETTE_SEED = 42

def build_kmeans_config(
    dataset,
    normalizacion,
    k,
    seed,
    inicializacion,
):
    """Construir la configuración completa de una ejecución."""

    discretizacion, n_puntos = dataset.rsplit("_", 1)

    return {
        "modelo": "kmeans",
        "dataset": dataset,
        "discretizacion": discretizacion,
        "n_puntos": int(n_puntos),
        "normalizacion": normalizacion,
        "inicializacion": inicializacion,
        "K": int(k),
        "seed_modelo": int(seed),
        "n_init": KMEANS_N_INIT,
        "max_iter": KMEANS_MAX_ITER,
        "tol": KMEANS_TOL,
        "algoritmo": KMEANS_ALGORITHM,
        "silhouette_n_muestras": SILHOUETTE_N_MUESTRAS,
        "silhouette_seed": SILHOUETTE_SEED,
        "version_sklearn": sklearn.__version__,
    }

def prepare_kmeans_data(dataset, normalizacion):
    """Cargar y normalizar los datos necesarios para una ejecución."""

    datos = cargar_dataset(dataset)

    X_train = aplicar_normalizacion(
        datos["train"]["X"],
        normalizacion,
    )

    X_val = aplicar_normalizacion(
        datos["val"]["X"],
        normalizacion,
    )

    metadata_train = datos["train"]["metadata"]
    metadata_val = datos["val"]["metadata"]

    return X_train, X_val, metadata_train, metadata_val

def train_kmeans_and_reconstruct(
    X_train,
    X_val,
    k,
    seed,
    inicializacion,
):
    """Entrenar K-Means y reconstruir entrenamiento y validación."""

    inicio = perf_counter()

    modelo = kmeans_model(
        X_train=X_train,
        k=k,
        seed=seed,
        inicializacion=inicializacion,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        tol=KMEANS_TOL,
        algorithm=KMEANS_ALGORITHM,
    )

    tiempo_entrenamiento = perf_counter() - inicio

    grupos_train, X_reconstruida_train = kmeans_reconstruction(
        modelo,
        X_train,
    )

    grupos_val, X_reconstruida_val = kmeans_reconstruction(
        modelo,
        X_val,
    )

    return {
        "modelo": modelo,
        "tiempo_entrenamiento": tiempo_entrenamiento,
        "grupos_train": grupos_train,
        "grupos_val": grupos_val,
        "X_reconstruida_train": X_reconstruida_train,
        "X_reconstruida_val": X_reconstruida_val,
    }

def calculate_kmeans_metrics(
    X_train,
    X_val,
    metadata_val,
    resultado_modelo,
):
    """Calcular métricas de entrenamiento y validación."""

    metricas = {
        "mse_train": mse_metric(
            X_train,
            resultado_modelo["X_reconstruida_train"],
        ),
        "rss_train": rss_metric(
            X_train,
            resultado_modelo["X_reconstruida_train"],
        ),
    }

    metricas_val = calcular_metricas_ejecucion(
        X=X_val,
        X_reconstruida=resultado_modelo["X_reconstruida_val"],
        subtipos=metadata_val["subtipo_rrlyrae"].to_numpy(),
        grupos=resultado_modelo["grupos_val"],
        silhouette_n_muestras=SILHOUETTE_N_MUESTRAS,
        silhouette_seed=SILHOUETTE_SEED,
    )

    for nombre, valor in metricas_val.items():
        metricas[f"{nombre}_val"] = valor

    return metricas

def build_kmeans_record(
    experiment_id,
    configuracion,
    metricas,
    resultado_modelo,
    X_train,
    X_val,
    metadata_train,
    metadata_val,
    carpeta_experimento,
    tiempo_total,
):
    """Reunir configuración, métricas y datos de ejecución."""

    registro = {
        "experiment_id": experiment_id,
        "fecha": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        **configuracion,
        "n_curvas_train": X_train.shape[0],
        "n_curvas_val": X_val.shape[0],
        "n_estrellas_train": metadata_train[
            "star_id_base"
        ].nunique(),
        "n_estrellas_val": metadata_val[
            "star_id_base"
        ].nunique(),
        "n_variables": X_train.shape[1],
        "tiempo_entrenamiento_segundos": resultado_modelo[
            "tiempo_entrenamiento"
        ],
        "tiempo_total_segundos": tiempo_total,
        "iteraciones_reales": resultado_modelo[
            "modelo"
        ].n_iter_,
        "alcanzo_max_iter": (
            resultado_modelo["modelo"].n_iter_
            >= configuracion["max_iter"]
        ),
        "estado": "OK",
        "ruta_experimento": str(
            carpeta_experimento.relative_to(PROJECT_ROOT)
        ),
        **metricas,
    }

    return registro

def run_kmeans_experiment(
    dataset,
    normalizacion,
    k,
    seed,
    inicializacion,
):
    """Ejecutar y guardar una configuración de K-Means."""

    inicio_total = perf_counter()

    configuracion = build_kmeans_config(
        dataset=dataset,
        normalizacion=normalizacion,
        k=k,
        seed=seed,
        inicializacion=inicializacion,
    )

    experiment_id = generate_experiment_id(configuracion)

    if experiment_is_registered(experiment_id):
        print(f"Experimento ya registrado: {experiment_id}")

        return {
            "experiment_id": experiment_id,
            "ejecutado": False,
        }

    X_train, X_val, metadata_train, metadata_val = (
        prepare_kmeans_data(
            dataset=dataset,
            normalizacion=normalizacion,
        )
    )

    resultado_modelo = train_kmeans_and_reconstruct(
        X_train=X_train,
        X_val=X_val,
        k=k,
        seed=seed,
        inicializacion=inicializacion,
    )

    metricas = calculate_kmeans_metrics(
        X_train=X_train,
        X_val=X_val,
        metadata_val=metadata_val,
        resultado_modelo=resultado_modelo,
    )

    carpeta_experimento = save_experiment_artifacts(
        experiment_id=experiment_id,
        configuracion=configuracion,
        patrones=resultado_modelo[
            "modelo"
        ].cluster_centers_,
        grupos_train=resultado_modelo["grupos_train"],
        grupos_val=resultado_modelo["grupos_val"],
        ids_train=metadata_train[
            "observacion_id"
        ].to_numpy(),
        ids_val=metadata_val[
            "observacion_id"
        ].to_numpy(),
    )

    tiempo_total = perf_counter() - inicio_total

    registro = build_kmeans_record(
        experiment_id=experiment_id,
        configuracion=configuracion,
        metricas=metricas,
        resultado_modelo=resultado_modelo,
        X_train=X_train,
        X_val=X_val,
        metadata_train=metadata_train,
        metadata_val=metadata_val,
        carpeta_experimento=carpeta_experimento,
        tiempo_total=tiempo_total,
    )

    save_experiment_record(registro)

    print(f"Experimento completado: {experiment_id}")
    print(
        "Tiempo de entrenamiento: "
        f"{registro['tiempo_entrenamiento_segundos']:.3f} segundos"
    )
    print(
        "Tiempo total: "
        f"{registro['tiempo_total_segundos']:.3f} segundos"
    )
    print(f"MSE validación: {registro['mse_val']:.6f}")

    return {
        "experiment_id": experiment_id,
        "ejecutado": True,
        "registro": registro,
    }

if __name__ == "__main__":
    resultado = run_kmeans_experiment(
        dataset="interpolacion_50",
        normalizacion="minmax",
        k=3,
        seed=1,
        inicializacion="k-means++",
    )

    print(resultado)