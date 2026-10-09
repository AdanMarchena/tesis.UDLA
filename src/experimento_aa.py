from datetime import datetime, timezone
from time import perf_counter

import archetypes
import numpy as np
import scipy
import sklearn

from exportacion import (
    PROJECT_ROOT,
    experiment_is_registered,
    generate_experiment_id,
    save_experiment_artifacts,
    save_experiment_record,
)
from loaders import cargar_dataset
from metricas import (
    calcular_metricas_ejecucion,
    mse_metric,
    rss_metric,
)
from modelos import (
    aa_model,
    aa_reconstruction,
)
from normalizacion import aplicar_normalizacion


AA_N_INIT = 1
AA_MAX_ITER = 300
AA_TOL = 1e-4
AA_OPTIMIZER = "nnls"

AA_NNLS_MAX_ITER = 1000
AA_NNLS_CONST = 100.0

SILHOUETTE_N_MUESTRAS = 5000
SILHOUETTE_SEED = 42


def build_aa_config(
    dataset,
    normalizacion,
    k,
    seed,
    inicializacion,
):
    """
    Construir la configuración completa
    de una ejecución de AA.
    """

    discretizacion, n_puntos = dataset.rsplit(
        "_",
        1,
    )

    return {
        "modelo": "archetypal_analysis",
        "dataset": dataset,
        "discretizacion": discretizacion,
        "n_puntos": int(n_puntos),
        "normalizacion": normalizacion,
        "inicializacion": inicializacion,
        "K": int(k),
        "seed_modelo": int(seed),
        "n_init": AA_N_INIT,
        "max_iter": AA_MAX_ITER,
        "tol": AA_TOL,
        "optimizador": AA_OPTIMIZER,
        "nnls_max_iter_optimizer": (
            AA_NNLS_MAX_ITER
        ),
        "nnls_const": AA_NNLS_CONST,
        "silhouette_n_muestras": (
            SILHOUETTE_N_MUESTRAS
        ),
        "silhouette_seed": SILHOUETTE_SEED,
        "version_sklearn": sklearn.__version__,
        "version_scipy": scipy.__version__,
        "version_archetypes": (
            archetypes.__version__
        ),
    }


def prepare_aa_data(
    dataset,
    normalizacion,
):
    """
    Cargar y normalizar entrenamiento
    y validación.
    """

    datos = cargar_dataset(dataset)

    X_train = aplicar_normalizacion(
        datos["train"]["X"],
        normalizacion,
    )

    X_val = aplicar_normalizacion(
        datos["val"]["X"],
        normalizacion,
    )

    metadata_train = (
        datos["train"]["metadata"]
    )

    metadata_val = (
        datos["val"]["metadata"]
    )

    return (
        X_train,
        X_val,
        metadata_train,
        metadata_val,
    )


def validate_aa_coefficients(
    coefficients,
    nombre,
    atol=1e-5,
):
    """
    Comprobar que los coeficientes
    pertenezcan al simplex.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            f"{nombre} debe ser una matriz "
            "bidimensional."
        )

    if not np.isfinite(
        coefficients
    ).all():
        raise ValueError(
            f"{nombre} contiene valores "
            "no finitos."
        )

    if np.any(
        coefficients < -atol
    ):
        raise ValueError(
            f"{nombre} contiene coeficientes "
            "negativos."
        )

    sumas = coefficients.sum(
        axis=1
    )

    if not np.allclose(
        sumas,
        1.0,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            f"Las filas de {nombre} "
            "no suman 1."
        )


def validate_aa_model(
    modelo,
    X_train,
    alphas_train,
    alphas_val,
):
    """
    Validar restricciones y resultados
    del modelo ajustado.
    """

    validate_aa_coefficients(
        alphas_train,
        "alphas_train",
    )

    validate_aa_coefficients(
        alphas_val,
        "alphas_val",
    )

    betas = np.asarray(
        modelo.arch_coefficients_,
        dtype=np.float64,
    )

    validate_aa_coefficients(
        betas,
        "betas",
    )

    arquetipos_calculados = (
        betas
        @ X_train
    )

    if not np.allclose(
        arquetipos_calculados,
        modelo.archetypes_,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Los arquetipos no coinciden "
            "con la combinación convexa "
            "beta @ X_train."
        )

    loss = np.asarray(
        modelo.loss_,
        dtype=np.float64,
    )

    if (
        loss.ndim != 1
        or loss.size == 0
    ):
        raise ValueError(
            "El modelo no entregó un historial "
            "de pérdida válido."
        )

    if not np.isfinite(
        loss
    ).all():
        raise ValueError(
            "El historial de pérdida contiene "
            "valores no finitos."
        )


def train_aa_and_reconstruct(
    X_train,
    X_val,
    k,
    seed,
    inicializacion,
):
    """
    Ajustar AA sobre las variables
    originales y reconstruir.
    """

    method_params = {
        "max_iter_optimizer": (
            AA_NNLS_MAX_ITER
        ),
        "const": AA_NNLS_CONST,
    }

    inicio = perf_counter()

    modelo = aa_model(
        X_train=X_train,
        k=k,
        seed=seed,
        inicializacion=inicializacion,
        n_init=AA_N_INIT,
        max_iter=AA_MAX_ITER,
        tol=AA_TOL,
        method=AA_OPTIMIZER,
        method_params=method_params,
    )

    tiempo_entrenamiento = (
        perf_counter()
        - inicio
    )

    alphas_train = np.asarray(
        modelo.coefficients_,
        dtype=np.float64,
    )

    (
        alphas_train,
        grupos_train,
        X_reconstruida_train,
    ) = aa_reconstruction(
        modelo=modelo,
        X=X_train,
        coefficients=alphas_train,
    )

    (
        alphas_val,
        grupos_val,
        X_reconstruida_val,
    ) = aa_reconstruction(
        modelo=modelo,
        X=X_val,
    )

    validate_aa_model(
        modelo=modelo,
        X_train=X_train,
        alphas_train=alphas_train,
        alphas_val=alphas_val,
    )

    betas = np.asarray(
        modelo.arch_coefficients_,
        dtype=np.float64,
    )

    arquetipos_originales = np.asarray(
        modelo.archetypes_,
        dtype=np.float64,
    )

    return {
        "modelo": modelo,
        "tiempo_entrenamiento": (
            tiempo_entrenamiento
        ),
        "grupos_train": grupos_train,
        "grupos_val": grupos_val,
        "X_reconstruida_train": (
            X_reconstruida_train
        ),
        "X_reconstruida_val": (
            X_reconstruida_val
        ),
        "alphas_train": alphas_train,
        "alphas_val": alphas_val,
        "betas": betas,
        "arquetipos_originales": (
            arquetipos_originales
        ),
    }


def calculate_aa_metrics(
    X_train,
    X_val,
    metadata_val,
    resultado_modelo,
):
    """
    Calcular las métricas de entrenamiento
    y validación.
    """

    metricas = {
        "mse_train": mse_metric(
            X_train,
            resultado_modelo[
                "X_reconstruida_train"
            ],
        ),
        "rss_train": rss_metric(
            X_train,
            resultado_modelo[
                "X_reconstruida_train"
            ],
        ),
    }

    metricas_val = (
        calcular_metricas_ejecucion(
            X=X_val,
            X_reconstruida=resultado_modelo[
                "X_reconstruida_val"
            ],
            subtipos=metadata_val[
                "subtipo_rrlyrae"
            ].to_numpy(),
            grupos=resultado_modelo[
                "grupos_val"
            ],
            silhouette_n_muestras=(
                SILHOUETTE_N_MUESTRAS
            ),
            silhouette_seed=(
                SILHOUETTE_SEED
            ),
        )
    )

    for (
        nombre,
        valor,
    ) in metricas_val.items():
        metricas[
            f"{nombre}_val"
        ] = valor

    return metricas


def build_aa_record(
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
    """
    Construir el registro
    de una ejecución de AA.
    """

    modelo = (
        resultado_modelo["modelo"]
    )

    registro = {
        "experiment_id": experiment_id,
        "fecha": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        **configuracion,
        "n_curvas_train": (
            X_train.shape[0]
        ),
        "n_curvas_val": (
            X_val.shape[0]
        ),
        "n_estrellas_train": (
            metadata_train[
                "star_id_base"
            ].nunique()
        ),
        "n_estrellas_val": (
            metadata_val[
                "star_id_base"
            ].nunique()
        ),
        "n_variables": (
            X_train.shape[1]
        ),
        "tiempo_entrenamiento_segundos": (
            resultado_modelo[
                "tiempo_entrenamiento"
            ]
        ),
        "tiempo_total_segundos": (
            tiempo_total
        ),
        "iteraciones_reales": (
            modelo.n_iter_
        ),
        "alcanzo_max_iter": (
            modelo.n_iter_
            >= configuracion["max_iter"]
        ),
        "convergio": (
            modelo.n_iter_
            < configuracion["max_iter"]
        ),
        "loss_final": float(
            modelo.loss_[-1]
        ),
        "estado": "OK",
        "ruta_experimento": str(
            carpeta_experimento.relative_to(
                PROJECT_ROOT
            )
        ),
        **metricas,
    }

    return registro


def run_aa_experiment(
    dataset,
    normalizacion,
    k,
    seed,
    inicializacion,
):
    """
    Ejecutar y guardar una configuración
    de AA.
    """

    inicio_total = perf_counter()

    configuracion = build_aa_config(
        dataset=dataset,
        normalizacion=normalizacion,
        k=k,
        seed=seed,
        inicializacion=inicializacion,
    )

    experiment_id = (
        generate_experiment_id(
            configuracion
        )
    )

    if experiment_is_registered(
        experiment_id
    ):
        print(
            "Experimento ya registrado: "
            f"{experiment_id}"
        )

        return {
            "experiment_id": experiment_id,
            "ejecutado": False,
        }

    (
        X_train,
        X_val,
        metadata_train,
        metadata_val,
    ) = prepare_aa_data(
        dataset=dataset,
        normalizacion=normalizacion,
    )

    resultado_modelo = (
        train_aa_and_reconstruct(
            X_train=X_train,
            X_val=X_val,
            k=k,
            seed=seed,
            inicializacion=inicializacion,
        )
    )

    metricas = calculate_aa_metrics(
        X_train=X_train,
        X_val=X_val,
        metadata_val=metadata_val,
        resultado_modelo=resultado_modelo,
    )

    carpeta_experimento = (
        save_experiment_artifacts(
            experiment_id=experiment_id,
            configuracion=configuracion,
            patrones=resultado_modelo[
                "arquetipos_originales"
            ],
            grupos_train=resultado_modelo[
                "grupos_train"
            ],
            grupos_val=resultado_modelo[
                "grupos_val"
            ],
            ids_train=metadata_train[
                "observacion_id"
            ].to_numpy(),
            ids_val=metadata_val[
                "observacion_id"
            ].to_numpy(),
            alphas_train=resultado_modelo[
                "alphas_train"
            ],
            alphas_val=resultado_modelo[
                "alphas_val"
            ],
            betas=resultado_modelo[
                "betas"
            ],
        )
    )

    tiempo_total = (
        perf_counter()
        - inicio_total
    )

    registro = build_aa_record(
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

    save_experiment_record(
        registro
    )

    print(
        "Experimento completado: "
        f"{experiment_id}"
    )

    print(
        "Tiempo de entrenamiento: "
        f"{registro['tiempo_entrenamiento_segundos']:.3f} "
        "segundos"
    )

    print(
        "Tiempo total: "
        f"{registro['tiempo_total_segundos']:.3f} "
        "segundos"
    )

    print(
        "RSS validación: "
        f"{registro['rss_val']:.6f}"
    )

    return {
        "experiment_id": experiment_id,
        "ejecutado": True,
        "registro": registro,
    }