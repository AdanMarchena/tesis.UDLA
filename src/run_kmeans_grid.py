from itertools import product

from experimento_kmeans import run_kmeans_experiment


DATASETS = (
    "interpolacion_50",
    "interpolacion_100",
    "medianas_fase_50",
    "medianas_fase_100",
)

NORMALIZACIONES = (
    "minmax",
    "centered",
    "zscore",
)

VALORES_K = tuple(range(2, 11))

SEMILLAS = (0, 1, 2, 3, 4)

INICIALIZACIONES = (
    "random",
    "k-means++",
)

def build_kmeans_grid():
    """Construir todas las configuraciones de K-Means."""

    combinaciones = product(
        DATASETS,
        NORMALIZACIONES,
        VALORES_K,
        SEMILLAS,
        INICIALIZACIONES,
    )

    configuraciones = []

    for (
        dataset,
        normalizacion,
        k,
        seed,
        inicializacion,
    ) in combinaciones:
        configuraciones.append({
            "dataset": dataset,
            "normalizacion": normalizacion,
            "k": k,
            "seed": seed,
            "inicializacion": inicializacion,
        })

    return configuraciones

def run_kmeans_grid(configuraciones, limite=None):
    """Ejecutar secuencialmente las configuraciones indicadas."""

    configuraciones_a_ejecutar = configuraciones[:limite]
    resultados = []

    total = len(configuraciones_a_ejecutar)

    for numero, configuracion in enumerate(
        configuraciones_a_ejecutar,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(f"Configuración {numero} de {total}")
        print(configuracion)
        print("=" * 60)

        resultado = run_kmeans_experiment(**configuracion)
        resultados.append(resultado)

    return resultados

if __name__ == "__main__":
    configuraciones = build_kmeans_grid()

    print(
        f"Configuraciones construidas: "
        f"{len(configuraciones):,}"
    )

    resultados = run_kmeans_grid(
        configuraciones=configuraciones,
        limite=None,
    )

    print(
        f"\nConfiguraciones procesadas en esta prueba: "
        f"{len(resultados)}"
    )