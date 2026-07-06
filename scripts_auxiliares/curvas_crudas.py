"""
curvas.py

Funciones de visualización para curvas de luz OGLE.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_lightcurve(
    file_path,
    show=True,
    save=False,
    output_path=None,
    figsize=(6, 4),
):
    """
    Grafica una curva de luz cruda: tiempo vs magnitud.

    Parameters
    ----------
    file_path : str or Path
        Ruta al archivo .dat de OGLE.
    show : bool
        Si True, muestra el gráfico.
    save : bool
        Si True, guarda la imagen.
    output_path : str or Path or None
        Ruta de salida. Si es None, guarda junto al nombre de la estrella.
    figsize : tuple
        Tamaño de la figura.

    Returns
    -------
    fig, ax
        Objetos de matplotlib.
    """

    file_path = Path(file_path)

    data = np.loadtxt(file_path)

    if data.ndim == 1:
        raise ValueError(f"El archivo tiene una sola observación: {file_path}")

    time = data[:, 0]
    mag = data[:, 1]

    star_id = file_path.stem

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(time, mag, s=5)
    ax.invert_yaxis()

    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Magnitud")
    ax.set_title(f"Curva cruda - {star_id}")
    ax.grid(alpha=0.3)

    if save:
        if output_path is None:
            output_path = file_path.with_suffix(".png")
        else:
            output_path = Path(output_path)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax