from experimento_aa import (
    run_aa_experiment,
)


DATASET_FINAL = "medianas_fase_50"
NORMALIZACION_FINAL = "minmax"
K_FINAL = 5

INICIALIZACION_FINAL = (
    "pca_convex_hull_furthest_sum"
)

SEMILLAS = (
    0,
    1,
    2,
    3,
    4,
)


def build_aa_grid():
    """
    Construir las cinco configuraciones
    finales de Análisis Arquetípico.
    """

    configuraciones = []

    for seed in SEMILLAS:
        configuraciones.append(
            {
                "dataset": DATASET_FINAL,
                "normalizacion": (
                    NORMALIZACION_FINAL
                ),
                "k": K_FINAL,
                "seed": seed,
                "inicializacion": (
                    INICIALIZACION_FINAL
                ),
            }
        )

    return configuraciones


def run_aa_grid(
    configuraciones,
):
    """
    Ejecutar secuencialmente las
    configuraciones finales de AA.
    """

    resultados = []

    total = len(
        configuraciones
    )

    for numero, configuracion in enumerate(
        configuraciones,
        start=1,
    ):
        print()
        print("=" * 60)

        print(
            "Configuración final de AA "
            f"{numero} de {total}"
        )

        print(configuracion)
        print("=" * 60)

        resultado = run_aa_experiment(
            **configuracion
        )

        resultados.append(
            resultado
        )

    return resultados


if __name__ == "__main__":
    configuraciones = build_aa_grid()

    print(
        "Configuraciones finales de "
        "Análisis Arquetípico: "
        f"{len(configuraciones)}"
    )

    resultados = run_aa_grid(
        configuraciones=configuraciones,
    )

    ejecutados = sum(
        resultado["ejecutado"]
        for resultado in resultados
    )

    omitidos = (
        len(resultados)
        - ejecutados
    )

    print()
    print("=" * 60)
    print("Ejecución finalizada")
    print(f"Experimentos nuevos: {ejecutados}")
    print(f"Experimentos omitidos: {omitidos}")
    print("=" * 60)