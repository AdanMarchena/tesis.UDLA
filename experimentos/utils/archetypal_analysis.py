"""
archetypal_analysis.py

Wrapper del modelo de Análisis Arquetípico.
"""

import contextlib
import io
import numpy as np
import re
import sys

from config import K, MAX_ITER, SEED
from utils.optimizacion import optimizar as _fit_model


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ArchetypalAnalysis:

    def __init__(self, K=K, max_iter=MAX_ITER, seed=SEED):

        self.K = K
        self.max_iter = max_iter
        self.seed = seed

        self.alpha = None
        self.beta = None
        self.Z = None
        self.convergio = False
        self.iteraciones_reales = 0
        self.error_final = None
        self.historial_error = []

    def fit(self, X):

        buffer = io.StringIO()

        with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
            resultado = _fit_model(
                X,
                K=self.K,
                max_iter=self.max_iter,
                seed=self.seed
            )

        historial_directo = None
        if isinstance(resultado, tuple) and len(resultado) == 4:
            self.alpha, self.beta, self.Z, historial_directo = resultado
        else:
            self.alpha, self.beta, self.Z = resultado

        salida = buffer.getvalue()
        if historial_directo is not None:
            self.historial_error = [float(error) for error in historial_directo]
        else:
            self.historial_error = [
                float(match.group(1))
                for match in re.finditer(r"Iter\s+\d+\s+\|\s+error:\s+([0-9.eE+-]+)", salida)
            ]
        self.iteraciones_reales = len(self.historial_error)
        self.convergio = (
            self.iteraciones_reales < self.max_iter
            if historial_directo is not None
            else "Convergencia alcanzada" in salida
        )
        self.error_final = float(np.linalg.norm(X - self.alpha @ self.Z, "fro"))

        return self
    
    def reconstruct(self):

        if self.alpha is None or self.Z is None:
            raise ValueError("Modelo no entrenado")

        return self.alpha @ self.Z
    
    def transform(self):

        if self.alpha is None:
            raise ValueError("Modelo no entrenado")

        return self.alpha
    
    def archetypes(self):

        if self.Z is None:
            raise ValueError("Modelo no entrenado")

        return self.Z
    
    def reconstruction_error(self, X):

        X_hat = self.reconstruct()
        return np.linalg.norm(X - X_hat, 'fro')
