"""
runner.py

Punto de entrada del experimento.
Orquesta carga de datos, entrenamiento, metricas y exportacion.
"""

import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import (
    mostrar_configuracion,
    PROJECT_ROOT,
    RESULTS_DIR,
    MODELO,
    K,
    MAX_ITER,
    SEED,
    DATASET,
    DISCRETIZACION,
    NORMALIZACION,
    INICIALIZACION,
    OPTIMIZACION,
    SUAVIZADO,
    N_PUNTOS,
    N_MUESTRAS,
    MODO_EJECUCION,
)

from utils.loaders import cargar_dataset
from utils.normalizacion import minmax, zscore
from utils.suavizado import suavizar
from utils.archetypal_analysis import ArchetypalAnalysis
from utils.metricas import evaluate_model
from utils.optimizacion import _obtener_nnls, project_to_simplex
from utils.exportacion import (
    config_serializable,
    crear_experimento,
    generar_id,
    actualizar_resumen_experimentos,
    guardar_config,
    guardar_metricas,
    guardar_modelo,
    preparar_experimento,
)
from utils.visualizacion import (
    plot_alpha_pca,
    plot_arquetipos_duplicados,
    plot_reconstrucciones,
    plot_simplex,
)

VERSION_ALGORITMO = "AA_alternating_v1"


def construir_config() -> dict:
    return {
        "modelo": MODELO,
        "dataset": DATASET,
        "discretizacion": DISCRETIZACION,
        "normalizacion": NORMALIZACION,
        "inicializacion": INICIALIZACION,
        "optimizacion": OPTIMIZACION,
        "suavizado": SUAVIZADO,
        "n_puntos": N_PUNTOS,
        "n_muestras": N_MUESTRAS,
        "K": K,
        "max_iter": MAX_ITER,
        "seed": SEED,
    }


def _primer_atributo(objeto, nombres):
    for nombre in nombres:
        if hasattr(objeto, nombre):
            return getattr(objeto, nombre)
    return None


def _metadata_estado(
    estado,
    convergio=None,
    iteraciones_reales=None,
    tiempo_por_iteracion=None,
    loss_final=None,
) -> dict:
    return {
        "estado": estado,
        "convergio": convergio,
        "iteraciones_reales": iteraciones_reales,
        "tiempo_por_iteracion": tiempo_por_iteracion,
        "version_algoritmo": VERSION_ALGORITMO,
        "loss_final": loss_final,
        "purity": None,
        "mean_activation": None,
        "silhouette": None,
        "davies_bouldin": None,
    }


def _normalizar_X(X):
    if NORMALIZACION == "minmax":
        return minmax(X)

    if NORMALIZACION == "zscore":
        return zscore(X)

    raise ValueError(f"Normalizacion no soportada en runner: {NORMALIZACION}")


def _calcular_alpha_nnls_con_Z(X, Z):
    nnls = _obtener_nnls()
    alpha = np.zeros((X.shape[0], Z.shape[0]))

    for i in range(X.shape[0]):
        a, _ = nnls(Z.T, X[i])
        alpha[i] = project_to_simplex(a)

    return alpha


def _metricas_reconstruccion(X, X_hat, prefijo):
    return {
        f"error_fro_{prefijo}": float(np.linalg.norm(X - X_hat, ord="fro")),
        f"error_rel_{prefijo}": float(
            np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-8)
        ),
    }


def _ruta_relativa(ruta):
    ruta = ruta.resolve()
    try:
        return str(ruta.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(ruta)


def _crear_carpetas_validacion(carpeta_exp):
    for subcarpeta in ("config", "metricas", "modelos", "figuras"):
        (carpeta_exp / subcarpeta).mkdir(parents=True, exist_ok=True)


def _duplicar_fase(y):
    y = np.asarray(y)
    fase = np.linspace(0.0, 1.0, y.shape[0], endpoint=False)
    return np.concatenate([fase, fase + 1.0]), np.concatenate([y, y])


def _plot_validacion_arquetipos(Z, ruta_figura):
    plt.figure(figsize=(9, 5))
    for k, z in enumerate(Z):
        fase, z_dup = _duplicar_fase(z)
        plt.plot(fase, z_dup, label=f"A{k + 1}")

    plt.xlabel("Fase")
    plt.ylabel("Intensidad")
    plt.title("Arquetipos duplicados")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=150)
    plt.close()


def _plot_validacion_reconstrucciones_test(X_test, X_hat_test, ids_test, ruta_figura):
    n = min(3, X_test.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n), squeeze=False)

    for fila, idx in enumerate(range(n)):
        ax = axes[fila, 0]
        fase, x_dup = _duplicar_fase(X_test[idx])
        _, x_hat_dup = _duplicar_fase(X_hat_test[idx])
        ax.plot(fase, x_dup, label="Original", linewidth=1.8)
        ax.plot(fase, x_hat_dup, label="Reconstruccion", linewidth=1.8)
        ax.set_title(str(ids_test[idx]))
        ax.set_xlabel("Fase")
        ax.set_ylabel("Intensidad")
        ax.legend()

    fig.tight_layout()
    fig.savefig(ruta_figura, dpi=150)
    plt.close(fig)


def _plot_validacion_proyeccion_alpha_test(alpha_test, ruta_figura):
    K_alpha = alpha_test.shape[1]
    plt.figure(figsize=(6, 5))

    if K_alpha == 3:
        x = alpha_test[:, 1] + 0.5 * alpha_test[:, 2]
        y = (np.sqrt(3) / 2) * alpha_test[:, 2]
        triangulo_x = [0, 1, 0.5, 0]
        triangulo_y = [0, 0, np.sqrt(3) / 2, 0]
        plt.plot(triangulo_x, triangulo_y, color="black", linewidth=1)
        plt.scatter(x, y, s=12, alpha=0.65)
        plt.axis("equal")
        plt.title("Simplex alpha test")
    else:
        coords = PCA(n_components=2).fit_transform(alpha_test)
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.65)
        plt.title("PCA alpha test")

    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=150)
    plt.close()


def _actualizar_resumen_validacion(fila):
    carpeta_validacion = RESULTS_DIR / "validacion_80_20"
    carpeta_validacion.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta_validacion / "resumen_validacion_80_20.csv"
    ruta_parquet = carpeta_validacion / "resumen_validacion_80_20.parquet"

    columnas = [
        "experiment_id",
        "fecha",
        "dataset",
        "discretizacion",
        "n_puntos",
        "normalizacion",
        "suavizado",
        "inicializacion",
        "optimizacion",
        "K",
        "n_muestras",
        "n_train",
        "n_test",
        "seed",
        "error_rel_train",
        "error_rel_test",
        "error_fro_train",
        "error_fro_test",
        "gap_generalizacion",
        "tiempo_segundos",
        "ruta_experimento",
    ]

    if ruta_csv.exists():
        resumen = pd.read_csv(ruta_csv)
        resumen = resumen[
            resumen["experiment_id"].astype(str) != str(fila["experiment_id"])
        ]
        resumen = pd.concat([resumen, pd.DataFrame([fila])], ignore_index=True)
    else:
        resumen = pd.DataFrame([fila])

    resumen = resumen[columnas]
    resumen.to_csv(ruta_csv, index=False)

    parquet_actualizado = True
    try:
        resumen.to_parquet(ruta_parquet, index=False)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        parquet_actualizado = False
        print(
            "No se pudo guardar resumen_validacion_80_20.parquet "
            f"({exc}). Se mantiene resumen_validacion_80_20.csv."
        )

    return {
        "csv": ruta_csv,
        "parquet": ruta_parquet,
        "csv_actualizado": True,
        "parquet_actualizado": parquet_actualizado,
    }


def run_train_test_experiment(test_size=0.2):
    inicio = time.perf_counter()
    config = construir_config()
    config_validacion = {
        **config,
        "validacion": "80_20",
        "test_size": float(test_size),
    }
    exp_id = generar_id(config_validacion)
    nombre_exp = f"{MODELO}_{exp_id}_{DATASET}_{DISCRETIZACION}_{N_PUNTOS}_K{K}"
    carpeta_exp = RESULTS_DIR / "validacion_80_20" / nombre_exp
    _crear_carpetas_validacion(carpeta_exp)

    print("\nValidacion train/test 80/20")
    print(f"ID validacion: {exp_id}")
    print(f"Carpeta      : {carpeta_exp}")

    X, ids = cargar_dataset(n_muestras=N_MUESTRAS, seed=SEED)
    X = np.asarray(X)
    ids = np.asarray(ids, dtype=str)
    X = _normalizar_X(X)
    X = suavizar(X)

    X_train, X_test, ids_train, ids_test = train_test_split(
        X,
        ids,
        test_size=test_size,
        random_state=SEED,
        shuffle=True,
    )

    model = ArchetypalAnalysis(K=K, max_iter=MAX_ITER, seed=SEED)
    model.fit(X_train)

    alpha_train = model.transform()
    Z_train = model.archetypes()
    X_hat_train = alpha_train @ Z_train

    alpha_test = _calcular_alpha_nnls_con_Z(X_test, Z_train)
    X_hat_test = alpha_test @ Z_train

    metrics = {
        **_metricas_reconstruccion(X_train, X_hat_train, "train"),
        **_metricas_reconstruccion(X_test, X_hat_test, "test"),
    }
    metrics["gap_generalizacion"] = (
        metrics["error_rel_test"] - metrics["error_rel_train"]
    )

    tiempo_segundos = time.perf_counter() - inicio

    with open(carpeta_exp / "config" / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_validacion, f, indent=4, ensure_ascii=False)

    with open(
        carpeta_exp / "metricas" / "metrics_train_test.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    carpeta_modelos = carpeta_exp / "modelos"
    np.save(carpeta_modelos / "Z.npy", Z_train)
    np.save(carpeta_modelos / "alpha_train.npy", alpha_train)
    np.save(carpeta_modelos / "alpha_test.npy", alpha_test)
    np.save(carpeta_modelos / "X_hat_train.npy", X_hat_train)
    np.save(carpeta_modelos / "X_hat_test.npy", X_hat_test)
    np.save(carpeta_modelos / "ids_train.npy", ids_train)
    np.save(carpeta_modelos / "ids_test.npy", ids_test)

    carpeta_figuras = carpeta_exp / "figuras"
    _plot_validacion_arquetipos(
        Z_train,
        carpeta_figuras / "arquetipos_duplicados.png",
    )
    _plot_validacion_reconstrucciones_test(
        X_test,
        X_hat_test,
        ids_test,
        carpeta_figuras / "reconstrucciones_test.png",
    )
    _plot_validacion_proyeccion_alpha_test(
        alpha_test,
        carpeta_figuras / "proyeccion_alpha_test.png",
    )

    fila = {
        "experiment_id": exp_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET,
        "discretizacion": DISCRETIZACION,
        "n_puntos": N_PUNTOS,
        "normalizacion": NORMALIZACION,
        "suavizado": SUAVIZADO,
        "inicializacion": INICIALIZACION,
        "optimizacion": OPTIMIZACION,
        "K": K,
        "n_muestras": N_MUESTRAS,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "seed": SEED,
        "error_rel_train": metrics["error_rel_train"],
        "error_rel_test": metrics["error_rel_test"],
        "error_fro_train": metrics["error_fro_train"],
        "error_fro_test": metrics["error_fro_test"],
        "gap_generalizacion": metrics["gap_generalizacion"],
        "tiempo_segundos": float(tiempo_segundos),
        "ruta_experimento": _ruta_relativa(carpeta_exp),
    }
    resumen = _actualizar_resumen_validacion(fila)

    print("\nValidacion finalizada.")
    print(f"error_rel_train: {metrics['error_rel_train']}")
    print(f"error_rel_test : {metrics['error_rel_test']}")
    print(f"gap            : {metrics['gap_generalizacion']}")
    print(f"Tiempo total   : {tiempo_segundos:.3f} segundos")

    return {
        "config": config_validacion,
        "id_experimento": exp_id,
        "carpeta_experimento": carpeta_exp,
        "metrics": metrics,
        "resumen": resumen,
        "tiempo_segundos": tiempo_segundos,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def run_final_experiment():
    inicio = time.perf_counter()
    config = {**construir_config(), "experimento_final": True}
    exp_id = generar_id(config)
    carpeta_exp = RESULTS_DIR / "experimento_final"
    X = None

    try:
        mostrar_configuracion()
        print("\nModo experimento final activado.")
        print(f"ID experimento final: {exp_id}")
        print(f"Carpeta             : {carpeta_exp}")

        crear_experimento(config, carpeta_exp)
        guardar_config(config, carpeta_exp)

        X, ids = cargar_dataset(n_muestras=N_MUESTRAS, seed=SEED)
        ids = np.asarray(ids, dtype=str)

        print("\nShape X:", X.shape)
        X = _normalizar_X(X)
        print("Normalizacion lista. Rango:", X.min(), X.max())
        X = suavizar(X)
        print(f"Suavizado aplicado: {SUAVIZADO}")

        model = ArchetypalAnalysis(
            K=K,
            max_iter=MAX_ITER,
            seed=SEED,
        )
        model.fit(X)

        alpha = model.transform()
        beta = model.beta if hasattr(model, "beta") else None
        Z = model.archetypes()
        X_hat = alpha @ Z
        metrics = evaluate_model(X, alpha, Z)

        tiempo_segundos = time.perf_counter() - inicio
        convergio = _primer_atributo(model, ("convergio", "converged", "converged_"))
        iteraciones_reales = _primer_atributo(
            model,
            ("iteraciones_reales", "n_iteraciones", "n_iter", "n_iter_", "iterations"),
        )
        tiempo_por_iteracion = (
            tiempo_segundos / iteraciones_reales if iteraciones_reales else None
        )
        loss_final = _primer_atributo(model, ("error_final",)) or metrics.get("error_fro")

        carpeta_metricas = carpeta_exp / "metricas"
        carpeta_metricas.mkdir(parents=True, exist_ok=True)
        with open(carpeta_metricas / "metrics.json", "w", encoding="utf-8") as archivo:
            json.dump(config_serializable(metrics), archivo, indent=4, ensure_ascii=False)

        guardar_modelo(config, alpha, Z, X_hat, beta=beta, carpeta=carpeta_exp)
        np.save(carpeta_exp / "modelos" / "ids.npy", ids)

        carpeta_figuras = carpeta_exp / "figuras"
        figuras = {
            "arquetipos": plot_arquetipos_duplicados(Z, carpeta_figuras),
            "reconstrucciones": plot_reconstrucciones(X, X_hat, ids, carpeta_figuras),
            "proyeccion": plot_simplex(alpha, carpeta_figuras)
            if K == 3
            else plot_alpha_pca(alpha, carpeta_figuras),
        }

        resumen = actualizar_resumen_experimentos(
            config=config,
            metrics=metrics,
            carpeta_exp=carpeta_exp,
            tiempo_segundos=tiempo_segundos,
            X=X,
            metadata=_metadata_estado(
                estado="OK",
                convergio=convergio,
                iteraciones_reales=iteraciones_reales,
                tiempo_por_iteracion=tiempo_por_iteracion,
                loss_final=loss_final,
            )
            | {"experimento_final": True},
        )

        from utils.validacion_astronomica import calcular_validacion_astronomica

        validacion = calcular_validacion_astronomica(
            carpeta_exp,
            guardar=True,
            salida_dir=carpeta_exp / "validacion_astronomica",
        )

        print("\nExperimento final exportado.")
        print(f"Estrellas usadas: {X.shape[0]}")
        print(f"error_rel       : {metrics['error_rel']}")
        print(f"convergio       : {convergio}")
        print(f"iteraciones     : {iteraciones_reales}")
        print(f"tiempo_segundos : {tiempo_segundos:.3f}")

        return {
            "config": config,
            "id_experimento": exp_id,
            "carpeta_experimento": carpeta_exp,
            "ejecutado": True,
            "estado": "OK",
            "ids": ids,
            "X": X,
            "alpha": alpha,
            "beta": beta,
            "Z": Z,
            "X_hat": X_hat,
            "metricas": metrics,
            "figuras": figuras,
            "resumen": resumen,
            "validacion_astronomica": validacion,
            "tiempo_segundos": tiempo_segundos,
            "convergio": convergio,
            "iteraciones_reales": iteraciones_reales,
            "tiempo_por_iteracion": tiempo_por_iteracion,
            "loss_final": loss_final,
            "historial_error": _primer_atributo(model, ("historial_error",)),
        }

    except MemoryError:
        tiempo_segundos = time.perf_counter() - inicio
        actualizar_resumen_experimentos(
            config=config,
            metrics={},
            carpeta_exp=carpeta_exp,
            tiempo_segundos=tiempo_segundos,
            X=X,
            metadata=_metadata_estado(estado="ERROR") | {"experimento_final": True},
        )
        print("Experimento final detenido limpiamente por falta de memoria.")
        raise


# ============================================================
# RUN EXPERIMENTO
# ============================================================

def run_experiment():
    inicio = time.perf_counter()
    config = construir_config()
    exp_id = generar_id(config)
    carpeta_exp = None
    X = None

    try:
        # ----------------------------------------------------
        # 1. Configuracion
        # ----------------------------------------------------
        mostrar_configuracion()

        # ----------------------------------------------------
        # 2. Preparar experimento
        # ----------------------------------------------------
        carpeta_exp, debe_ejecutar = preparar_experimento(
            config,
            modo_ejecucion=MODO_EJECUCION,
        )

        if not debe_ejecutar:
            print("\nExperimento ya existe. Ejecucion omitida.")
            print(f"ID experimento: {exp_id}")
            print(f"Carpeta       : {carpeta_exp}")
            return {
                "config": config,
                "id_experimento": exp_id,
                "carpeta_experimento": carpeta_exp,
                "ejecutado": False,
            }

        print(f"\nID experimento: {exp_id}")
        print(f"Carpeta       : {carpeta_exp}")

        guardar_config(config, carpeta_exp)

        # ----------------------------------------------------
        # 3. Cargar datos
        # ----------------------------------------------------
        X, ids = cargar_dataset(n_muestras=N_MUESTRAS, seed=SEED)

        print("\nShape X:", X.shape)

        # ----------------------------------------------------
        # 4. Normalizacion
        # ----------------------------------------------------
        X = _normalizar_X(X)

        print("Normalizacion lista. Rango:", X.min(), X.max())

        X = suavizar(X)
        print(f"Suavizado aplicado: {SUAVIZADO}")

        # ----------------------------------------------------
        # 5. Entrenamiento
        # ----------------------------------------------------
        model = ArchetypalAnalysis(
            K=K,
            max_iter=MAX_ITER,
            seed=SEED,
        )

        model.fit(X)

        alpha = model.transform()
        beta = model.beta if hasattr(model, "beta") else None
        Z = model.archetypes()

        X_hat = alpha @ Z

        print("\nModelo entrenado:")
        print("alpha:", alpha.shape)
        print("beta:", None if beta is None else beta.shape)
        print("Z:", Z.shape)
        print("X_hat:", X_hat.shape)

        # ----------------------------------------------------
        # 6. Metricas
        # ----------------------------------------------------
        metrics = evaluate_model(X, alpha, Z)

        print("\nMETRICAS:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # ----------------------------------------------------
        # 7. Exportacion
        # ----------------------------------------------------
        tiempo_segundos = time.perf_counter() - inicio
        convergio = _primer_atributo(model, ("convergio", "converged", "converged_"))
        iteraciones_reales = _primer_atributo(
            model,
            ("iteraciones_reales", "n_iteraciones", "n_iter", "n_iter_", "iterations"),
        )
        tiempo_por_iteracion = (
            tiempo_segundos / iteraciones_reales if iteraciones_reales else None
        )
        loss_final = _primer_atributo(model, ("error_final",)) or metrics.get("error_fro")

        guardar_metricas(config, metrics, carpeta_exp)
        guardar_modelo(config, alpha, Z, X_hat, beta=beta, carpeta=carpeta_exp)

        carpeta_figuras = carpeta_exp / "figuras"
        figuras = {
            "arquetipos": plot_arquetipos_duplicados(Z, carpeta_figuras),
            "reconstrucciones": plot_reconstrucciones(X, X_hat, ids, carpeta_figuras),
            "proyeccion": plot_simplex(alpha, carpeta_figuras)
            if K == 3
            else plot_alpha_pca(alpha, carpeta_figuras),
        }

        resumen = actualizar_resumen_experimentos(
            config=config,
            metrics=metrics,
            carpeta_exp=carpeta_exp,
            tiempo_segundos=tiempo_segundos,
            X=X,
            metadata=_metadata_estado(
                estado="OK",
                convergio=convergio,
                iteraciones_reales=iteraciones_reales,
                tiempo_por_iteracion=tiempo_por_iteracion,
                loss_final=loss_final,
            ),
        )

        print("\nExportacion finalizada.")
        print(f"Tiempo total: {tiempo_segundos:.3f} segundos")

        resultado = {
            "config": config,
            "id_experimento": exp_id,
            "carpeta_experimento": carpeta_exp,
            "ejecutado": True,
            "estado": "OK",
            "ids": ids,
            "X": X,
            "alpha": alpha,
            "beta": beta,
            "Z": Z,
            "X_hat": X_hat,
            "metricas": metrics,
            "figuras": figuras,
            "resumen": resumen,
            "tiempo_segundos": tiempo_segundos,
            "convergio": convergio,
            "iteraciones_reales": iteraciones_reales,
            "tiempo_por_iteracion": tiempo_por_iteracion,
            "loss_final": loss_final,
            "historial_error": _primer_atributo(model, ("historial_error",)),
        }

        return resultado

    except KeyboardInterrupt:
        # Punto preparado para registrar estado="INTERRUMPIDO" cuando se habilite
        # manejo explicito de interrupciones manuales.
        raise

    except Exception:
        tiempo_segundos = time.perf_counter() - inicio
        if carpeta_exp is not None:
            actualizar_resumen_experimentos(
                config=config,
                metrics={},
                carpeta_exp=carpeta_exp,
                tiempo_segundos=tiempo_segundos,
                X=X,
                metadata=_metadata_estado(estado="ERROR"),
            )
        raise


# ============================================================
# EJECUCION DIRECTA
# ============================================================

if __name__ == "__main__":
    run_experiment()
