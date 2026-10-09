import numpy as np

def minmax_normalization(X):
    """Escalar cada curva al intervalo [0, 1]"""
    X = np.asarray(X, dtype=np.float64)

    minimo = X.min(axis=1, keepdims=True)
    maximo = X.max(axis=1, keepdims=True)
    rango = maximo - minimo

    if np.any(rango == 0):
        raise ValueError("Hay curvas constantes: no se puede aplicar min-max.")

    return (X-minimo) / rango

def centered_normalization(X):
    """Centrar cada curva restando su propia media."""
    X = np.asarray(X, dtype=np.float64)

    media = X.mean(axis=1, keepdims=True)

    return X - media


def zscore_normalization(X):
    """Centrar cada curva y escalarla a desviación estándar 1"""
    X = np.asarray(X, dtype=np.float64)

    media = X.mean(axis=1, keepdims=True)
    desviacion = X.std(axis=1, keepdims=True, ddof=0)

    if np.any(desviacion == 0):
        raise ValueError("Hay curvas constantes: no se puede aplicar z-score.")
    
    return (X - media) / desviacion

def aplicar_normalizacion(X, metodo):
    """APlicar la normalización indicada."""
    metodos = {
        "minmax": minmax_normalization,
        "centered": centered_normalization,
        "zscore": zscore_normalization,
    }
    if metodo not in metodos:
        raise ValueError(f"Metodo de normalización no reconocido: {metodo}")

    return metodos[metodo](X)

if __name__ == "__main__":
    X_ejemplo = np.array([
        [1, 2, 3],
        [10, 20, 30],
    ])

    for metodo in ("minmax", "centered", "zscore"):
        resultado = aplicar_normalizacion(X_ejemplo, metodo)
        print(f"\n{metodo}:")
        print(resultado)