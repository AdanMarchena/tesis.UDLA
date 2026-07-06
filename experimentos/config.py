"""
config.py

Configuración general de los experimentos.
"""

from pathlib import Path

# ==========================================================
# Rutas del proyecto
# ==========================================================

# Carpeta raíz de experimentos
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "datos_discretizados"
RESULTS_DIR = PROJECT_ROOT / "resultados"

# ==========================================================
# Modelo
# ==========================================================

MODELOS = ("AA", "KMEANS", "ADA")

MODELO = "AA"

# ==========================================================
# Parámetros del experimento
# ==========================================================

# Valores permitidos
DATASETS = ("rrlyrae", "cefeidas")
DISCRETIZACIONES = ("interpolacion", "percentiles")
NORMALIZACIONES = ("minmax", "zscore")
INICIALIZACIONES = ("random", "pca", "furthest_sum")
OPTIMIZACIONES = ("alternating", "pgd", "nnls", "robust_pgd")
SUAVIZADOS = ("none", "savgol")

DATASET = "rrlyrae"
DISCRETIZACION = "percentiles"
NORMALIZACION = "minmax"
INICIALIZACION = "random"
OPTIMIZACION = "nnls"
SUAVIZADO = "none"

# ==========================================================
# Dataset
# ==========================================================

# 50 o 100 puntos

N_PUNTOS = 50

# None = usar todo el dataset
N_MUESTRAS = None

# ==========================================================
# Modelo
# ==========================================================

# Número de arquetipos o clusters

K = 6

MAX_ITER = 100

SEED = 42

# ==========================================================
# Hiperparametros PGD
# ==========================================================

LR_ALPHA = 1e-3
LR_BETA = 1e-5
N_ALPHA_STEPS = 20
N_BETA_STEPS = 10
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2
PATIENCE = 20
ALPHA_ITER = 40
BETA_ITER = 20
ROBUST_LR_BETA = 1e-4

# ==========================================================
# Gestión de experimentos
# ==========================================================

# skip
# overwrite
# new

MODO_EJECUCION = "skip"

# ==========================================================
# Visualización
# ==========================================================

GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True

# ==========================================================
# Utilidades
# ==========================================================

def mostrar_configuracion():

    print("=" * 60)
    print("CONFIGURACIÓN DEL EXPERIMENTO")
    print("=" * 60)

    print(f"Modelo             : {MODELO}")
    print(f"Dataset            : {DATASET}")
    print(f"Discretización     : {DISCRETIZACION}")
    print(f"N° puntos          : {N_PUNTOS}")
    print(f"N° muestras        : {N_MUESTRAS}")
    print(f"Normalización      : {NORMALIZACION}")
    print(f"Inicialización     : {INICIALIZACION}")
    print(f"Optimización       : {OPTIMIZACION}")
    print(f"Suavizado          : {SUAVIZADO}")
    print(f"K                  : {K}")
    print(f"Iteraciones máximas: {MAX_ITER}")
    print(f"Semilla            : {SEED}")
    print(f"Modo ejecución     : {MODO_EJECUCION}")

    print("=" * 60)
