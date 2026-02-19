import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import time
np . random . seed (42)
# Cargar datos
data = pd.read_csv("creditcard.csv")

# Exploración básica

# Calcular frecuencias absolutas
frequencies_absolute = data['Class'].value_counts() 

# Calcular frecuencias relativas
frequencies_relative = data['Class'].value_counts(normalize=True) * 100

# Imprimir tabla
print("Tabla de Frecuencias de la Variable 'Class':")
print("------------------------------------------------")
print("Clase | Frecuencia Absoluta | Frecuencia Relativa (%)")
print("------------------------------------------------")
for clase in frequencies_absolute.index:
    print(f"{clase:^5} | {frequencies_absolute[clase]:^19} | {frequencies_relative[clase]:^23.2f}")
print("------------------------------------------------")

# Gráfico de barras a escala logaritmica
clases = ['No Fraude (0)', 'Fraude (1)']

plt.figure(figsize=(8, 6))
plt.bar(clases, frequencies_absolute, color=['skyblue', 'orange'], edgecolor='k', log=True)
plt.title("Frecuencia Absoluta (Escala Logarítmica)")
plt.ylabel("Frecuencia Absoluta (Log)")
plt.xlabel("Clase")
for i, v in enumerate(frequencies_absolute):
    plt.text(i, v, f"{v}", ha='center', va='bottom') 
plt.show()

# Se separan la variable objetivo del resto de la base
X = data.drop(columns=['Class'])
y = data['Class']


# Preprocesamiento: Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar Frequent Directions usando randomized_svd

# Definir el número de componentes (dimensiones) tras reducción
n_components = 10

# Aplicar sketching usando SVD aleatorizado
U, Sigma, VT = randomized_svd(X_train, n_components=n_components, random_state=42)
X_train_sketch = np.dot(U, np.diag(Sigma))

# También aplicamos la misma transformación en X_test
X_test_sketch = np.dot(X_test, VT.T)

#Comparación de la dimensionalidad
print(f"Dimensiones originales: {X_train.shape}")
print(f"Dimensiones reducidas: {X_train_sketch.shape}")

# Temporizador para el modelo con datos originales
start_time = time.time()  # Iniciar el temporizador

# Entrenar modelo con datos originales
model_original = LogisticRegression(random_state=42)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

end_time = time.time()  # Finalizar el temporizador
clasificacion_original = classification_report(y_test, y_pred_original)
print("Resultados con datos originales:")
print(clasificacion_original)
print(f"Tiempo de entrenamiento y predicción con datos originales: {end_time - start_time:.4f} segundos")
print("\n")

# Temporizador para el modelo con datos reducidos
start_time2 = time.time()  # Iniciar el temporizador

# Entrenar modelo con datos reducidos
model_sketch = LogisticRegression(random_state=42)
model_sketch.fit(X_train_sketch, y_train)
y_pred_sketch = model_sketch.predict(X_test_sketch)

end_time2 = time.time()  # Finalizar el temporizador
clasificacion_sketch = classification_report(y_test, y_pred_sketch)
print("Resultados con Frequent Directions:")
print(clasificacion_sketch)
print(f"Tiempo de entrenamiento y predicción con Frequent Directions: {end_time2 - start_time2:.4f} segundos")

# Comparación de los modelos

# Primero pasamos las clasificaciones a un diccionario para asi poder seleccionar mejor el elemnto que queremos
clasificacion_sketch = classification_report(y_test, y_pred_sketch,output_dict=True)
clasificacion_original = classification_report(y_test, y_pred_original,output_dict=True)

# Gráficos de comparación con etiquetas de valores
metrics = ['accuracy', 'precision', 'recall', 'f1-score']
original_scores = [
    clasificacion_original['accuracy'],
    clasificacion_original['1']['precision'],
    clasificacion_original['1']['recall'],
    clasificacion_original['1']['f1-score']
]
sketch_scores = [
    clasificacion_sketch['accuracy'],
    clasificacion_sketch['1']['precision'],
    clasificacion_sketch['1']['recall'],
    clasificacion_sketch['1']['f1-score']
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, original_scores, width, label='Original', color='blue')
bars2 = ax.bar(x + width/2, sketch_scores, width, label='Frequent Directions', color='orange')

# Añadir etiquetas en las barras
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Comparación de Rendimiento')
ax.legend()
plt.show()

# Tabla comparativa
print("\nTabla Comparativa de Resultados:")
print("----------------------------------------------------------------------------------------------------")
print(f"{'Modelo':^25} | {'Tiempo de Ejecución (s)':^20} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10}")
print("----------------------------------------------------------------------------------------------------")

# Mostrar los resultados para el modelo original
print(f"{'Original':^25} | {end_time - start_time:^23.4f} | {original_scores[0]:^10.4f} | {original_scores[1]:^10.4f} | {original_scores[2]:^10.4f} | {original_scores[3]:^10.4f}")

# Mostrar los resultados para el modelo reducido (Frequent Directions)
print(f"{'Frequent Directions':^25} | {end_time2 - start_time2:^23.4f} | {sketch_scores[0]:^10.4f} | {sketch_scores[1]:^10.4f} | {sketch_scores[2]:^10.4f} | {sketch_scores[3]:^10.4f}")

print("----------------------------------------------------------------------------------------------------")

# Gráfico de dispersión de las primeras dos direcciones principales del espacio reducido
plt.figure(figsize=(8, 6))
plt.scatter(X_train_sketch[:, 0], X_train_sketch[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel("Primera dirección principal")
plt.ylabel("Segunda dirección principal")
plt.title("Datos en espacio de sketching (Frequent Directions)")
plt.colorbar(label="Fraude (0: No, 1: Sí)")
plt.show()


