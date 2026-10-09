import archetypes
import numpy as np

from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def kmeans_model(
    X_train,
    k,
    seed,
    inicializacion,
    n_init=1,
    max_iter=300,
    tol=1e-4,
    algorithm="lloyd",
):
    """Ajustar K-Means únicamente con entrenamiento."""

    modelo = KMeans(
        n_clusters=k,
        init=inicializacion,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        algorithm=algorithm,
    )

    modelo.fit(X_train)

    return modelo


def kmeans_reconstruction(modelo, X):
    """Asignar grupos y reconstruir curvas con centroides."""

    X = np.asarray(X, dtype=np.float64)

    grupos = modelo.predict(X)
    X_reconstruida = modelo.cluster_centers_[grupos]

    return grupos, X_reconstruida


def pca_furthest_sum_initialization_indices(
    X_train,
    k,
    seed,
):
    """
    Seleccionar K curvas mediante PCA, Convex Hull
    y Furthest Sum.
    """

    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    if X_train.ndim != 2:
        raise ValueError(
            "X_train debe ser una matriz bidimensional."
        )

    n_muestras, n_variables = X_train.shape

    if k > n_muestras:
        raise ValueError(
            f"K ({k}) no puede superar el número "
            f"de curvas ({n_muestras})."
        )

    if min(n_muestras, n_variables) < 2:
        raise ValueError(
            "La inicialización PCA necesita al menos "
            "dos muestras y dos variables."
        )

    pca = PCA(
        n_components=2,
        random_state=seed,
    )

    X_pca = pca.fit_transform(X_train)

    try:
        hull = ConvexHull(X_pca)
        indices_candidatos = hull.vertices
    except QhullError:
        indices_candidatos = np.arange(
            n_muestras
        )

    if len(indices_candidatos) < k:
        indices_candidatos = np.arange(
            n_muestras
        )

    candidatos_pca = X_pca[
        indices_candidatos
    ]

    rng = np.random.default_rng(seed)

    primera_posicion = int(
        rng.integers(len(indices_candidatos))
    )

    posiciones_seleccionadas = [
        primera_posicion
    ]

    suma_distancias = np.linalg.norm(
        candidatos_pca
        - candidatos_pca[primera_posicion],
        axis=1,
    )

    while len(posiciones_seleccionadas) < k:
        suma_distancias[
            posiciones_seleccionadas
        ] = -np.inf

        siguiente_posicion = int(
            np.argmax(suma_distancias)
        )

        posiciones_seleccionadas.append(
            siguiente_posicion
        )

        nuevas_distancias = np.linalg.norm(
            candidatos_pca
            - candidatos_pca[siguiente_posicion],
            axis=1,
        )

        mascara_disponibles = np.isfinite(
            suma_distancias
        )

        suma_distancias[
            mascara_disponibles
        ] += nuevas_distancias[
            mascara_disponibles
        ]

    indices_seleccionados = (
        indices_candidatos[
            posiciones_seleccionadas
        ]
    )

    return np.asarray(
        indices_seleccionados,
        dtype=np.int64,
    )


class PCAFurthestSumInitializedAA(
    archetypes.AA
):
    """
    AA con inicialización PCA + Convex Hull
    + Furthest Sum.
    """

    def _init_archetypes(self, X, rng):
        n_muestras = X.shape[0]

        indices = (
            pca_furthest_sum_initialization_indices(
                X_train=X,
                k=self.n_archetypes,
                seed=self.random_state,
            )
        )

        B = np.zeros(
            (
                self.n_archetypes,
                n_muestras,
            ),
            dtype=X.dtype,
        )

        B[
            np.arange(self.n_archetypes),
            indices,
        ] = 1.0

        arquetipos = X[indices].copy()

        A = np.zeros(
            (
                n_muestras,
                self.n_archetypes,
            ),
            dtype=X.dtype,
        )

        asignaciones_iniciales = rng.choice(
            self.n_archetypes,
            n_muestras,
            replace=True,
        )

        A[
            np.arange(n_muestras),
            asignaciones_iniciales,
        ] = 1.0

        return A, B, arquetipos


def aa_model(
    X_train,
    k,
    seed,
    inicializacion,
    n_init=1,
    max_iter=300,
    tol=1e-4,
    method="nnls",
    method_params=None,
):
    """
    Ajustar análisis arquetípico con entrenamiento.
    """

    parametros = {
        "n_archetypes": k,
        "n_init": n_init,
        "max_iter": max_iter,
        "tol": tol,
        "method": method,
        "method_params": method_params,
        "random_state": seed,
        "save_init": True,
        "verbose": False,
    }

    inicializaciones_pca = {
        "pca_furthest_sum",
        "pca_convex_hull_furthest_sum",
    }

    if inicializacion in inicializaciones_pca:
        modelo = PCAFurthestSumInitializedAA(
            init="uniform",
            **parametros,
        )

    elif inicializacion in {
        "uniform",
        "furthest_sum",
    }:
        modelo = archetypes.AA(
            init=inicializacion,
            **parametros,
        )

    else:
        raise ValueError(
            "Inicialización de AA no soportada: "
            f"{inicializacion}"
        )

    modelo.fit(X_train)

    return modelo


def aa_reconstruction(
    modelo,
    X,
    coefficients=None,
):
    """
    Calcular coeficientes, grupos y reconstrucciones.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    if coefficients is None:
        coefficients = modelo.transform(X)

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    grupos = np.argmax(
        coefficients,
        axis=1,
    )

    X_reconstruida = (
        coefficients @ modelo.archetypes_
    )

    return (
        coefficients,
        grupos,
        X_reconstruida,
    )


if __name__ == "__main__":
    X_prueba = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [6.0, 6.0],
    ])

    for inicializacion in (
        "random",
        "k-means++",
    ):
        modelo_kmeans = kmeans_model(
            X_train=X_prueba,
            k=2,
            seed=42,
            inicializacion=inicializacion,
        )

        grupos, X_reconstruida = (
            kmeans_reconstruction(
                modelo_kmeans,
                X_prueba,
            )
        )

        assert grupos.shape == (8,)
        assert (
            X_reconstruida.shape
            == X_prueba.shape
        )
        assert (
            modelo_kmeans.cluster_centers_.shape
            == (2, 2)
        )

    indices_pca = (
        pca_furthest_sum_initialization_indices(
            X_train=X_prueba,
            k=3,
            seed=42,
        )
    )

    assert indices_pca.shape == (3,)
    assert len(np.unique(indices_pca)) == 3
    assert np.all(indices_pca >= 0)
    assert np.all(
        indices_pca < len(X_prueba)
    )

    print(
        "Las pruebas de modelos e inicialización "
        "PCA + Convex Hull + Furthest Sum "
        "finalizaron correctamente."
    )

