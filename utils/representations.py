import numpy as np
from scipy.interpolate import interp1d

# Función para generar representación vectorial mediante percentiles

def percentiles_repr(phase, mag, M=100):

    bins = np.linspace(0, 1, M+1)
    x_i = []

    for i in range(M):
        mask = (phase >= bins[i]) & (phase < bins[i+1])

        if np.any(mask):
            value = np.median(mag[mask])
        else:
            value = np.nan

        x_i.append(value)

    x_i = np.array(x_i)

    # rellenar NaN
    if np.any(np.isnan(x_i)):
        valid = ~np.isnan(x_i)
        x_i = np.interp(
            np.arange(M),
            np.where(valid)[0],
            x_i[valid]
        )

    return x_i


# Función para generar representación vectorial mediante interpolación

def interpolation_repr(phase, mag, M=100):
    """
    Genera representación vectorial mediante interpolación.

    Parámetros
    ----------
    phase : np.ndarray
    mag : np.ndarray
    M : int
        Número de puntos en la grilla

    Retorna
    -------
    x_i : np.ndarray
        Vector de tamaño M
    """

    # grilla uniforme
    grid = np.linspace(0, 1, M)

    # interpolador
    f = interp1d(
        phase,
        mag,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    x_i = f(grid)

    return x_i