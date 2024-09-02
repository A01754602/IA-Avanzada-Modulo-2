# IA-Avanzada-Modulo-2

Este repositorio contiene el código para el proyecto del Módulo 2 de IA Avanzada.

## Descripción

Este proyecto implementa un modelo de regresión lineal para predecir el tiempo de vuelta de carros de carreras basándose en su peso. Posteriormente, el modelo clasifica los vehículos en “Carro de carreras” o “No es carro de carreras” comparando el tiempo de vuelta predicho con el tiempo real. La evaluación se realiza utilizando una matriz de confusión.

## Requisitos

- **Python**: Versión 3.7 o superior
- **Bibliotecas**:
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
- **Datos**: Un archivo Excel con los datos necesarios, que contenga las siguientes columnas:
  - **Peso (kg)**: Peso del carro
  - **Tiempo de vuelta (s)**: Tiempo de vuelta del carro
  - **Carro de carreras**: Etiqueta que indica si es un carro de carreras

## Estructura del Código

1. **Importación de bibliotecas y carga de datos**: El código comienza importando las bibliotecas necesarias y cargando los datos desde un archivo Excel.
2. **Preprocesamiento de datos**: Se extraen las características, el objetivo, y las etiquetas del DataFrame, y se dividen los datos en conjuntos de entrenamiento y prueba.
3. **Descenso de gradiente para regresión lineal**: Se implementa un algoritmo de descenso de gradiente para ajustar una línea a los datos de entrenamiento.
4. **Clasificación y evaluación**: Utilizando la línea ajustada, se clasifica cada carro y se evalúa el rendimiento del modelo mediante una matriz de confusión.

## Instrucciones de Uso

1. **Clonar el repositorio**: Clona este repositorio en tu máquina local.
2. **Instalar dependencias**: Ejecuta `pip install -r requirements.txt` para instalar todas las dependencias necesarias.
3. **Ejecutar el código**: Ejecuta el archivo principal en tu entorno Python para realizar la predicción y clasificación.
4. **Ver los resultados**: Los resultados se mostrarán en la terminal y la matriz de confusión se visualizará mediante una gráfica.

## Posibles Mejoras

- Implementar validación cruzada para evaluar mejor el modelo.
- Ajustar automáticamente los hiperparámetros (alpha y tolerance) utilizando técnicas de optimización.
- Ampliar el análisis utilizando métricas adicionales como la curva ROC y el AUC.

