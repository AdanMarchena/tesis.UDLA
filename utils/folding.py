import numpy as np

def fold_curve(time, mag, period, t0):

    time = np.asarray(time)
    mag = np.asarray(mag)

    if len(time) == 0 or len(mag) == 0:
        raise ValueError("Arrays vacíos")

    if len(time) != len(mag):
        raise ValueError("time y mag deben tener la misma longitud")

    if period <= 0:
        raise ValueError("El período debe ser positivo")

    phase = ((time - t0) / period) % 1

    idx = np.argsort(phase)
    phase = phase[idx]
    mag = mag[idx]

    phase, unique_idx = np.unique(phase, return_index=True)
    mag = mag[unique_idx]

    if len(phase) < 5:
        raise ValueError("Muy pocos puntos")

    return phase, mag