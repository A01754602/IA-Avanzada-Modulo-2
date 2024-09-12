# IA-Avanzada-Modulo-2

Este repositorio contiene el código para el proyecto del Módulo 2 de IA Avanzada.

## Descripción

Este proyecto implementa modelos de regresión para predecir el tiempo de vuelta de carros de carreras basándose en su peso. Además, el proyecto clasifica los vehículos en “Carro de carreras” o “No es carro de carreras” mediante una comparación entre el tiempo de vuelta predicho y el tiempo real, utilizando tanto **Regresión Lineal (Ridge)** como **Regresión Logística** para la clasificación. Se evalúan los modelos utilizando matrices de confusión, métricas de regresión y clasificación.

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

1. **Importación de bibliotecas y carga de datos**: Se importan las bibliotecas necesarias y se cargan los datos desde un archivo Excel.
2. **Preprocesamiento de datos**: Se extraen las características, el objetivo y las etiquetas del DataFrame, dividiendo los datos en conjuntos de entrenamiento y prueba. Se normalizan las variables para optimizar el rendimiento de los modelos.
3. **Modelos de regresión**:
   - **Regresión Lineal (Ridge)**: Se implementa un modelo de regresión lineal con regularización (Ridge) para predecir el tiempo de vuelta basado en el peso del carro.
   - **Regresión Logística**: Se utiliza un modelo de regresión logística para clasificar los carros en función de si son "Carro de carreras" o "No es carro de carreras".
4. **Evaluación de los modelos**:
   - Para **Ridge**, se calculan métricas como el **Error Cuadrático Medio (MSE)** y el **Coeficiente de Determinación (R²)**, además de visualizar las predicciones en comparación con los datos reales.
   - Para **Regresión Logística**, se utilizan métricas como **Precisión**, **Recall**, **F1-Score** y **Precisión Total**, junto con una visualización de la **matriz de confusión** para el conjunto de prueba.
5. **Visualización**: Se crean gráficas para mostrar las predicciones de cada modelo y las matrices de confusión, junto con las métricas clave de desempeño.

## Instrucciones de Uso

1. **Clonar el repositorio**: Clona este repositorio en tu máquina local.
2. **Instalar dependencias**: Ejecuta `pip install -r requirements.txt` para instalar todas las dependencias necesarias.
3. **Ejecutar el código**: Ejecuta el archivo principal en tu entorno Python para realizar la predicción, clasificación y ver las evaluaciones de los modelos.
4. **Ver los resultados**: Los resultados se mostrarán en la terminal y las visualizaciones, incluidas las matrices de confusión y las gráficas de predicción, se generarán mediante Matplotlib.

## Posibles Mejoras

- Implementar validación cruzada para evaluar mejor el modelo.
- Ajustar automáticamente los hiperparámetros (alpha y tolerance) utilizando técnicas de optimización.
- Ampliar el análisis utilizando métricas adicionales como la curva ROC y el AUC para la regresión logística.
- Añadir modelos adicionales como **Random Forest** o **XGBoost** para comparar el rendimiento con los modelos actuales.