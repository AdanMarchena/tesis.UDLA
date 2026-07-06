"""
folding.py

Funciones para aplicar folding a curvas de luz.

El folding transforma una curva observada en tiempo a una curva en fase,
usando el período de variabilidad y una época de referencia.
"""

from __future__ import annotations

import numpy as np


def fold_curve(time, mag, period, t0):
    """
    Aplica folding a una curva de luz.

    Parameters
    ----------
    time : array-like
        Tiempos de observación.
    mag : array-like
        Magnitudes observadas.
    period : float
        Período de la estrella.
    t0 : float
        Época de referencia.

    Returns
    -------
    phase : np.ndarray
        Fases ordenadas en el intervalo [0, 1).
    mag : np.ndarray
        Magnitudes ordenadas según la fase.

    Raises
    ------
    ValueError
        Si los datos están vacíos, tienen longitudes distintas,
        el período no es positivo o quedan muy pocos puntos.
    """

    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if time.size == 0 or mag.size == 0:
        raise ValueError("time y mag no pueden estar vacíos.")

    if time.shape[0] != mag.shape[0]:
        raise ValueError("time y mag deben tener la misma longitud.")

    if period <= 0:
        raise ValueError("El período debe ser positivo.")

    phase = ((time - t0) / period) % 1

    idx = np.argsort(phase)
    phase = phase[idx]
    mag = mag[idx]

    phase, unique_idx = np.unique(phase, return_index=True)
    mag = mag[unique_idx]

    if phase.size < 5:
        raise ValueError("Muy pocos puntos después del folding.")

    return phase, mag


def duplicate_phase_curve(phase, mag):
    """
    Duplica una curva foldeada para visualizar dos ciclos.

    Parameters
    ----------
    phase : array-like
        Fases en [0, 1).
    mag : array-like
        Magnitudes asociadas.
    
    Returns
    -------
    phase_dup : np.ndarray
        Fases duplicadas en [0, 2).
    mag_dup : np.ndarray
        Magnitudes duplicadas.
    """

    phase = np.asarray(phase, dtype=float)
    mag = np.asarray(mag, dtype=float)

    phase_dup = np.concatenate([phase, phase + 1])
    mag_dup = np.concatenate([mag, mag])

    return phase_dup, mag_dup