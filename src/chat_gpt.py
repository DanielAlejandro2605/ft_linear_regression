import numpy as np
import math

from modules.plotting import plot_with_regression_line

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression.
    """
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw += (f_wb - y[i]) * x[i] 
        dj_db += (f_wb - y[i])

    dj_dw /= m 
    dj_db /= m 
        
    return dj_dw, dj_db

def gradient_descent(data_x, data_y, initial_w=0, initial_b=0, learning_rate=0.01, num_iters=1000, tolerance=1e-7):
    # Normalización de los datos para evitar problemas de escala
    x_mean = np.mean(data_x)
    print(x_mean)
    x_std = np.std(data_x)
    normalized_x = (data_x - x_mean) / x_std

    # Inicialización de los coeficientes
    w = initial_w
    b = initial_b
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(normalized_x, data_y, w , b)

        # Verificar si los gradientes son `nan` o `inf`
        if math.isnan(dj_dw) or math.isnan(dj_db) or math.isinf(dj_dw) or math.isinf(dj_db):
            print("Overflow detected! Stopping gradient descent.")
            break

        # Actualizar los coeficientes
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        # Calcular y registrar el costo para verificar la convergencia
        cost = np.mean((w * normalized_x + b - data_y) ** 2) / 2
        cost_history.append(cost)

        # Condición de parada basada en la magnitud del gradiente (tolerancia)
        if abs(dj_dw) < tolerance and abs(dj_db) < tolerance:
            print(f"Converged after {i} iterations.")
            break

    return w, b, cost_history

# Dataset sencillo
data_x = np.array([100, 200, 300])
data_y = np.array([1000, 2000, 3000])

# Ejecutar el algoritmo
w_final, b_final, costs = gradient_descent(data_x, data_y, learning_rate=0.01)

print(f"Final w: {w_final}, Final b: {b_final}")

plot_with_regression_line(data_x, data_y, w_final, b_final)
