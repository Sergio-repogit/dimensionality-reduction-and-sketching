# Dimensionality Reduction and Sketching

Repositorio centrado en técnicas de reducción de dimensionalidad, sketching matricial y compresión de datos de alta dimensión.  
Los proyectos incluidos exploran aplicaciones prácticas en detección de fraude, resolución aproximada de sistemas de mínimos cuadrados y compresión de imágenes.

---

## Contenido del repositorio

### fraud-detection-frequent-directions.py
Implementación del algoritmo **Frequent Directions** aplicada a detección de fraude en transacciones.

**Incluye:**
- `fraud-detection-frequent-directions.py` — script principal  
- memoria asociada al proyecto  
- dataset `creditcard`  

**Objetivo:**  
Reducir la dimensionalidad del conjunto de transacciones preservando información relevante para análisis de fraude.

---

### randomized-least-squares-countsketch
Implementación de **Randomized Least Squares** utilizando **CountSketch**.

**Incluye:**
- script del algoritmo  
- memoria asociada  

**Objetivo:**  
Aproximar soluciones de mínimos cuadrados en alta dimensión mediante técnicas de sketching eficientes.

---

### image_compression_randomized_svd_rgb
Compresión de imágenes en color mediante **Randomized SVD**.

**Incluye:**
- script de compresión  
- memoria del proyecto  
- imagen `jinxarcane-dibujos-neon-flotantes-4fjnl0z0y5cfdmbl`  

**Objetivo:**  
Reducir el rango de la imagen RGB manteniendo la mayor calidad visual posible y analizando el coste computacional.

---

## Técnicas implementadas

- Frequent Directions  
- CountSketch  
- Randomized Least Squares  
- Randomized SVD  
- Compresión matricial  
- Reducción de dimensionalidad  

---

## Tecnologías utilizadas

- Python  
- NumPy  
- SciPy  
- Matplotlib  
- Scikit-learn  

---
