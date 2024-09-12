import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Importar el set de datos para su correcto manejo
df = pd.read_excel('dataset_carros.xlsx')

""" 
Se dividen los datos en dos conjuntos, uno de entrenamiento y otro de prueba.
Se selecciona el peso del carro como variable independiente y el tiempo de vuelta como variable dependiente.
Se separan los datos en 70% para entrenamiento y 30% para prueba.
"""
x = np.array(df['Peso (kg)'])
y = np.array(df['Tiempo de vuelta (s)'])
carro = np.array(df['Carro de carreras'])

# Separar los datos en entrenamiento y prueba.
x_train, x_test, y_train, y_test, carro_train, carro_test = train_test_split(x, y, carro, test_size=0.3, random_state=42)

# Normalización de los datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train.reshape(-1, 1)).flatten()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

x_test_scaled = scaler_x.transform(x_test.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

"""
Definimos los parámetros iniciales para el modelo de regresión.
"""

# Parámetros iniciales para el modelo de regresión
m = 0
b = 0
epochs = 10000
alpha = 0.01
tolerance = 1e-6
n_t = len(x_train_scaled)

"""
Algoritmo de descenso de gradiente.
"""
for epoch in range(epochs):
    # Calculo de la prediccion y el error
    y_pred = m * x_train_scaled + b
    error = y_pred - y_train_scaled
    # Calculo del gradiente 
    grad_m = (1/n_t) * sum(error * x_train_scaled)
    grad_b = (1/n_t) * sum(error)
    # Actualización de los parámetros
    m = m - alpha * grad_m
    b = b - alpha * grad_b
    # Verificar la convergencia
    if abs(alpha * grad_m) < tolerance and abs(alpha * grad_b) < tolerance:
        print('Converge en la iteracion:', epoch)
        break
    # Imprimir valores de los parámetros cada 100 iteraciones
    if epoch % 100 == 0:
        print(f'Iteración: {epoch+1}, m = {m}, b = {b}')

# Resultados del modelo en los datos de entrenamiento
print(f'La pendiente es: {m}')
print(f'La intersección es: {b}')

# Predicciones en los datos de entrenamiento normalizados
y_train_pred_scaled = m * x_train_scaled + b

# Desescalar las predicciones y valores originales
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_train_original = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()

# Evaluación del modelo con métricas más adecuadas
mse = mean_squared_error(y_train_original, y_train_pred)
r2 = r2_score(y_train_original, y_train_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# Clasificador
clasificador_carr_t = np.where(y_train_original > y_train_pred, "Sí", "No")
matriz = confusion_matrix(carro_train, clasificador_carr_t, labels=["Sí", "No"])
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Carro de carreras", "No es carro de carreras"])

disp = disp.plot(cmap=plt.cm.Blues)
plt.title("Carros")
plt.show()

# Precision, recall y f1-score
precision = precision_score(carro_train, clasificador_carr_t, pos_label="Sí")
recall = recall_score(carro_train, clasificador_carr_t, pos_label="Sí")
f1 = f1_score(carro_train, clasificador_carr_t, pos_label="Sí")
print(f'Precisión: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(x_train[carro_train == 'Sí'], y_train[carro_train == 'Sí'], color='red', label='Carro de carreras')
plt.scatter(x_train[carro_train == 'No'], y_train[carro_train == 'No'], color='blue', label='No es carro de carreras')
plt.scatter(x_train[clasificador_carr_t == 'Sí'], y_train[clasificador_carr_t == 'Sí'], color='green', marker='x', label='Clasificado como carro de carreras')
plt.scatter(x_train[clasificador_carr_t == 'No'], y_train[clasificador_carr_t == 'No'], color='purple', marker='x', label='Clasificado como no carro de carreras')
plt.plot(x_train, y_train_pred, color='black', label='Recta de regresión')

plt.xlabel('Peso')
plt.ylabel('Tiempo de vuelta')
plt.title(f'Regresión Lineal con Clasificador\nMSE: {mse:.4f}, R²: {r2:.4f}')
plt.legend()
plt.grid(True)
plt.show()