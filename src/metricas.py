import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

# MÉTRICA MSE
def mse_metric(X, X_reconstruida):
    """Calcular el error cuadrático medio por valor de la matriz."""
    X = np.asarray(X, dtype=np.float64)
    X_reconstruida = np.asarray(X_reconstruida, dtype=np.float64)

    if X.shape != X_reconstruida.shape:
        raise ValueError(
            "Las matrices original y reconstruida deben tener "
            "las mismas dimensiones."
        )

    return float(np.mean((X - X_reconstruida) ** 2))

# MÉTRICA RSS
def rss_metric(X, X_reconstruida):
    """Calcular la suma de errores cuadrados de reconstrucción."""
    X = np.asarray(X, dtype=np.float64)
    X_reconstruida = np.asarray(X_reconstruida, dtype=np.float64)

    if X.shape != X_reconstruida.shape:
        raise ValueError(
            "Las matrices original y reconstruida deben tener "
            "las mismas dimensiones."
        )

    return float(np.sum((X - X_reconstruida) ** 2))

# MÉTRICA PUREZA
def purity_metric(subtipos, grupos):
    """Calcular la pureza de los grupos respecto a los subtipos."""
    subtipos = np.asarray(subtipos)
    grupos = np.asarray(grupos)

    if subtipos.ndim != 1 or grupos.ndim != 1:
        raise ValueError("Los subtipos y grupos deben ser vectores.")

    if len(subtipos) != len(grupos) or len(subtipos) == 0:
        raise ValueError(
            "Debe haber un subtipo y un grupo por curva, "
            "con al menos una curva."
        )

    if pd.isna(subtipos).any() or pd.isna(grupos).any():
        raise ValueError("Los subtipos y grupos no pueden contener faltantes.")

    tabla = pd.crosstab(
        index=subtipos,
        columns=grupos,
    )

    mayoritarios = tabla.max(axis=0)

    return float(mayoritarios.sum() / len(subtipos))

# MÉTRICA DAVIES-BOULDIN
def davies_bouldin_metric(X, grupos):
    """Evaluar la compactación y separación de los grupos."""
    X = np.asarray(X, dtype=np.float64)
    grupos = np.asarray(grupos)

    if X.ndim != 2 or grupos.ndim != 1:
        raise ValueError(
            "X debe ser una matriz y grupos debe ser un vector."
        )

    if X.shape[0] != len(grupos):
        raise ValueError("Debe haber un grupo asignado por curva.")

    if not np.isfinite(X).all() or pd.isna(grupos).any():
        raise ValueError("Los datos y grupos contienen valores no válidos.")

    n_grupos = len(np.unique(grupos))

    if not 2 <= n_grupos < X.shape[0]:
        raise ValueError(
            "Davies–Bouldin requiere al menos dos grupos "
            "y menos grupos que curvas."
        )

    return float(davies_bouldin_score(X, grupos))

# MÉTRICA SILHOUETTE
def silhouette_metric(X, grupos, n_muestras=5000, seed=42):
    """Calcular silhouette sobre una muestra reproducible de curvas."""
    X = np.asarray(X, dtype=np.float64)
    grupos = np.asarray(grupos)

    if X.ndim != 2 or grupos.ndim != 1:
        raise ValueError(
            "X debe ser una matriz y grupos debe ser un vector."
        )

    if X.shape[0] != len(grupos):
        raise ValueError("Debe haber un grupo asignado por curva.")

    if not np.isfinite(X).all() or pd.isna(grupos).any():
        raise ValueError("Los datos y grupos contienen valores no válidos.")

    if n_muestras is not None:
        if not isinstance(n_muestras, (int, np.integer)) or n_muestras < 3:
            raise ValueError("n_muestras debe ser un entero de al menos 3.")

        if X.shape[0] > n_muestras:
            rng = np.random.default_rng(seed)
            indices = rng.choice(
                X.shape[0], size=n_muestras, replace=False
            )
            X = X[indices]
            grupos = grupos[indices]

    n_grupos = len(np.unique(grupos))

    if not 2 <= n_grupos < X.shape[0]:
        raise ValueError(
            "La muestra evaluada debe contener al menos dos grupos "
            "y menos grupos que curvas."
        )

    return float(silhouette_score(X, grupos, metric="euclidean"))

# MÉTRICA AMI
def ami_metric(subtipos, grupos):
    """Medir la asociación entre subtipos y grupos, ajustada por azar."""
    subtipos = np.asarray(subtipos)
    grupos = np.asarray(grupos)

    if subtipos.ndim != 1 or grupos.ndim != 1:
        raise ValueError("Los subtipos y grupos deben ser vectores.")

    if len(subtipos) != len(grupos) or len(subtipos) == 0:
        raise ValueError(
            "Debe haber un subtipo y un grupo por curva, "
            "con al menos una curva."
        )

    if pd.isna(subtipos).any() or pd.isna(grupos).any():
        raise ValueError("Los subtipos y grupos no pueden contener faltantes.")

    return float(
        adjusted_mutual_info_score(
            subtipos,
            grupos,
            average_method="arithmetic",
        )
    )

# MÉTRICA ARI
def ari_metric(grupos_a, grupos_b):
    """Comparar las asignaciones de grupos de dos ejecuciones."""
    grupos_a = np.asarray(grupos_a)
    grupos_b = np.asarray(grupos_b)

    if grupos_a.ndim != 1 or grupos_b.ndim != 1:
        raise ValueError("Ambas asignaciones deben ser vectores.")

    if len(grupos_a) != len(grupos_b) or len(grupos_a) == 0:
        raise ValueError(
            "Debe haber dos asignaciones para las mismas curvas."
        )

    if pd.isna(grupos_a).any() or pd.isna(grupos_b).any():
        raise ValueError("Las asignaciones no pueden contener faltantes.")

    return float(adjusted_rand_score(grupos_a, grupos_b))

def calcular_metricas_ejecucion(
    X,
    X_reconstruida,
    subtipos,
    grupos,
    silhouette_n_muestras=5000,
    silhouette_seed=42,
):
    """Calcular las métricas correspondientes a una ejecución."""

    metricas = {
        "mse": mse_metric(X, X_reconstruida),
        "rss": rss_metric(X, X_reconstruida),
        "purity": purity_metric(subtipos, grupos),
        "davies_bouldin": davies_bouldin_metric(X, grupos),
        "silhouette": silhouette_metric(
            X,
            grupos,
            n_muestras=silhouette_n_muestras,
            seed=silhouette_seed,
        ),
        "ami": ami_metric(subtipos, grupos),
    }

    return metricas

if __name__ == "__main__":
    X_prueba = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [10.0, 10.0],
        [11.0, 11.0],
    ])

    X_reconstruida = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [10.5, 10.5],
        [10.5, 10.5],
    ])

    subtipos = np.array([
        "RRab",
        "RRab",
        "RRc",
        "RRc",
    ])

    grupos_a = np.array([0, 0, 1, 1])
    grupos_b = np.array([1, 1, 0, 0])

    metricas_ejecucion = calcular_metricas_ejecucion(
        X=X_prueba,
        X_reconstruida=X_reconstruida,
        subtipos=subtipos,
        grupos=grupos_a,
        silhouette_n_muestras=None,
    )

    print("\nMétricas reunidas:")
    print(metricas_ejecucion)

    resultados = {
        "mse": mse_metric(X_prueba, X_reconstruida),
        "rss": rss_metric(X_prueba, X_reconstruida),
        "purity": purity_metric(subtipos, grupos_a),
        "davies_bouldin": davies_bouldin_metric(
            X_prueba, grupos_a
        ),
        "silhouette": silhouette_metric(
            X_prueba,
            grupos_a,
            n_muestras=None,
        ),
        "ami": ami_metric(subtipos, grupos_a),
        "ari": ari_metric(grupos_a, grupos_b),
    }

    for nombre, valor in resultados.items():
        print(f"{nombre}: {valor:.6f}")

    assert np.isclose(resultados["mse"], 0.25)
    assert np.isclose(resultados["rss"], 2.0)
    assert np.isclose(resultados["purity"], 1.0)
    assert np.isclose(resultados["ami"], 1.0)
    assert np.isclose(resultados["ari"], 1.0)

    print("\nTodas las comprobaciones finalizaron correctamente.")