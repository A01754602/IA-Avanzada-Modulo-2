import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Importar el set de datos
df = pd.read_excel('dataset_carros.xlsx')

# Definir las variables
x = np.array(df['Peso (kg)']).reshape(-1, 1)
y = np.array(df['Tiempo de vuelta (s)'])
carro = np.array(df['Carro de carreras'])

# Convertir 'Carro de carreras' en binario (para regresión logística)
y_clasificacion = np.where(carro == 'Sí', 1, 0)

# Separar los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test, y_train_clasif, y_test_clasif = train_test_split(x, y, y_clasificacion, test_size=0.3, random_state=42)

# Normalización de los datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

### Modelo 1: Regresión Lineal (Ridge)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train_scaled, y_train_scaled)
y_train_pred_ridge = ridge_model.predict(x_train_scaled)
y_test_pred_ridge = ridge_model.predict(x_test_scaled)

# Desescalar las predicciones
y_train_pred_ridge_orig = scaler_y.inverse_transform(y_train_pred_ridge.reshape(-1, 1)).flatten()
y_test_pred_ridge_orig = scaler_y.inverse_transform(y_test_pred_ridge.reshape(-1, 1)).flatten()

# Evaluación del modelo Ridge
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge_orig)
r2_train_ridge = r2_score(y_train, y_train_pred_ridge_orig)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge_orig)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge_orig)

### Visualización de los resultados de Ridge

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Datos reales')
plt.plot(x_train, y_train_pred_ridge_orig, color='red', label='Predicciones Ridge')
plt.title(f'Regresión Ridge\nMSE en prueba: {mse_test_ridge:.4f}, R² en prueba: {r2_test_ridge:.4f}')
plt.xlabel('Peso')
plt.ylabel('Tiempo de vuelta')
plt.legend()
plt.grid(True)
plt.show()

### Modelo 2: Regresión Logística

log_reg = LogisticRegression()
log_reg.fit(x_train_scaled, y_train_clasif)

# Predicciones para regresión logística
y_train_pred_log = log_reg.predict(x_train_scaled)
y_test_pred_log = log_reg.predict(x_test_scaled)

# Evaluación del modelo Logístico (conjunto de entrenamiento)
accuracy_train_log = accuracy_score(y_train_clasif, y_train_pred_log)
precision_train_log = precision_score(y_train_clasif, y_train_pred_log)
recall_train_log = recall_score(y_train_clasif, y_train_pred_log)
f1_train_log = f1_score(y_train_clasif, y_train_pred_log)

# Evaluación del modelo Logístico (conjunto de prueba)
accuracy_test_log = accuracy_score(y_test_clasif, y_test_pred_log)
precision_test_log = precision_score(y_test_clasif, y_test_pred_log)
recall_test_log = recall_score(y_test_clasif, y_test_pred_log)
f1_test_log = f1_score(y_test_clasif, y_test_pred_log)

# Matriz de confusión para el conjunto de prueba (Regresión Logística)
matriz_conf_log = confusion_matrix(y_test_clasif, y_test_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=matriz_conf_log, display_labels=['No', 'Sí'])
disp_log.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión - Regresión Logística (Prueba)")
plt.show()

# Matriz de confusión para el conjunto de entrenamiento (Regresión Logística)
matriz_conf_train_log = confusion_matrix(y_train_clasif, y_train_pred_log)
disp_train_log = ConfusionMatrixDisplay(confusion_matrix=matriz_conf_train_log, display_labels=['No', 'Sí'])
disp_train_log.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión - Regresión Logística (Entrenamiento)")
plt.show()

# Resultados de los modelos:

# Resultados de Ridge
print(f"Resultados de Regresión Ridge:\n - MSE en entrenamiento: {mse_train_ridge}\n - R² en entrenamiento: {r2_train_ridge}")
print(f" - MSE en prueba: {mse_test_ridge}\n - R² en prueba: {r2_test_ridge}")

# Resultados de Regresión Logística
print(f"Resultados de Regresión Logística - Entrenamiento:\n - Precisión: {accuracy_train_log}\n - Precision: {precision_train_log}\n - Recall: {recall_train_log}\n - F1: {f1_train_log}")
print(f"Resultados de Regresión Logística - Prueba:\n - Precisión: {accuracy_test_log}\n - Precision: {precision_test_log}\n - Recall: {recall_test_log}\n - F1: {f1_test_log}")