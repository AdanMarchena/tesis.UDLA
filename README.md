# Archetypal Analysis for Variable Stars

Este repositorio contiene el código desarrollado para la tesis de Magíster en Ciencia de Datos de la Universidad de Las Américas (UDLA).

El proyecto estudia la aplicación de **Archetypal Analysis (AA)** al análisis de curvas de luz de estrellas variables provenientes del proyecto **OGLE**, con énfasis en estrellas **RR Lyrae**.

## Objetivos

- Construir un pipeline reproducible para el análisis de curvas de luz.
- Evaluar distintas estrategias de inicialización y optimización para Archetypal Analysis.
- Comparar AA con métodos de referencia como K-Means.
- Analizar la relación entre los arquetipos obtenidos y los subtipos astronómicos de RR Lyrae.

## Estructura del repositorio

```text
scripts_auxiliares/
    Scripts utilizados durante la preparación del dataset.

experimentos/
    config.py
    runner.py
    notebooks/
    utils/

README.md
requirements.txt
LICENSE
```

## Dataset

Los datos utilizados provienen del proyecto **OGLE (Optical Gravitational Lensing Experiment)**.

Debido a las políticas de distribución y al tamaño de los datos, los archivos originales no se incluyen en este repositorio.

## Estado del proyecto

Actualmente el proyecto se encuentra en la etapa final de experimentación y redacción de la tesis.

Las siguientes etapas incluyen:

- Finalización del manuscrito de tesis.
- Preparación de un artículo científico.
- Desarrollo de una aplicación para el análisis de estrellas variables.
- Desarrollo de una librería reutilizable para Archetypal Analysis.

## Autor

Adán Marchena

Magíster en Ciencia de Datos

Universidad de Las Américas (UDLA)

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
