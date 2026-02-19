

import numpy as np
import matplotlib.pyplot as plt
# Simulamos un conjunto de datos financieros

np . random . seed (42)

m = 1000000 # Numero de transacciones
n = 20 # Numero de caracteristicas (monto , ubicacion , metodo de pago , etc .)

# Generamos una matriz A de caracteristicas aleatorias
A = np.random.rand(m,n)
# Generamos un vector de etiquetas de fraude (1 para fraude , 0 para legitima )
b = (np.random.rand( m ) > 0.98).astype(int) # 2% de fraudes simulados


print ("Dimensiones de la matriz A:", A . shape )
print ("Dimensiones del vector b:", b . shape )


def count_sketch (A , sketch_size ) :
    m,n = A.shape
    S = np.zeros(( sketch_size , m ) )
    # Construimos la matriz dispersa CountSketch
    for i in range ( m ) :
        row_index = np.random.randint(0 , sketch_size )
        sign = np.random.choice([1 , -1])
        S[row_index,i] = sign
    # Aplicamos la proyeccion
    A_sketch = S @ A
    b_sketch = S @ b
    return A_sketch , b_sketch



sketch_size = 3000 #Tamano reduciodo
A_skecth,b_sketch=count_sketch(A,sketch_size)

print (" Dimensiones de la matriz A tras la reduccion de dimensionalidad:", A_skecth . shape )
print (" Dimensiones del vector b tras la reduccion de dimensionalidad:", b_sketch . shape )

#Resolucion del problema red
from numpy.linalg import qr

#Primero vamos a resolver el problema con las matrices despues de la reduccion de la dimensionalidad
#Descomposicion QR de la matriz reducida A_sketch
Q,R = qr(A_skecth)

#Resolucion del sistema de minimos cuadrados en el espacio reducido
x_sketch = np.linalg.solve(R, Q.T @ b_sketch)  ##Soluciona un sistema de la forma Rx= Q.T@b_sketch


##Aqui vamos a solucionar el problema con los valores reales.
##Aqui esta el problema principal que no es realista calcular el valor sin la reduccion de la dimensionalidad.
##A la hora de probar los metodos esta bien plantarselo bajo unos parametros conocidos (ejercicios con tamano muy grande pero que ya se haya calculado la solucion de manera previa)
Q_real,R_real=qr(A)

X_real=np.linalg.solve(R_real, Q_real.T @ b)  ##Soluciona un sistema de la forma Rx= Q.T@b_sketch

#Mostramos la solucion aproximada en el espacio reducido
print("Solucion aproximada en el espacio reducido: ",x_sketch)


#Calculamos el error relativo
error=np.linalg.norm(X_real-x_sketch)/np.linalg.norm(X_real)


#Mostramos el error relativo
print(f"error relativo en la estimacion de parametros : {error:.4f}")

#Visualizacion de los parametros reales vs estimados
plt.figure(figsize=(10,6))
plt.plot(X_real,label="Parametros reales",marker='o')
plt.plot(x_sketch,label="Parametros estimados",marker='x')
plt.title("Comparacion de parametros reales y estimados")
plt.xlabel("Indice de parametro")
plt.ylabel("Valor del paramtro")
plt.legend()
plt.show()




