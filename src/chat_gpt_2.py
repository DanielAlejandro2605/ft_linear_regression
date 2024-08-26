import numpy as np
import math
import pandas as pd

def compute_gradient(x, y, w, b): 
    # Implementación del cálculo de gradientes (igual que antes)
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
    # Normalización de los datos
    x_mean = np.mean(data_x)
    print(x_mean)
    x_std = np.std(data_x)
    print(x_std)
    normalized_x = (data_x - x_mean) / x_std

    # Inicialización de los coeficientes
    w = initial_w
    b = initial_b
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(normalized_x, data_y, w , b)

        if math.isnan(dj_dw) or math.isnan(dj_db) or math.isinf(dj_dw) or math.isinf(dj_db):
            print("Overflow detected! Stopping gradient descent.")
            break

        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        cost = np.mean((w * normalized_x + b - data_y) ** 2) / 2
        cost_history.append(cost)

        if abs(dj_dw) < tolerance and abs(dj_db) < tolerance:
            print(f"Converged after {i} iterations.")
            break

    w_original = w / x_std
    b_original = b - (w * x_mean / x_std)

    return w_original, b_original, cost_history

# Reading the file
df = pd.read_csv('../data/data_example.csv')
# Getting km data from dataframe
data_x = df['km'].to_numpy()
# Getting price data from dataframe
data_y = df['price'].to_numpy()

# Ejecutar el algoritmo
w_final, b_final, costs = gradient_descent(data_x, data_y, learning_rate=0.01)

print(f"Final w: {w_final}, Final b: {b_final}")

# Graficar los datos y la línea de regresión
import matplotlib.pyplot as plt

plt.scatter(data_x, data_y, color='blue', marker='x', label='Data Points')
plt.plot(data_x, w_final * data_x + b_final, color='red', label=f'Regression Line: y = {w_final:.2f}x + {b_final:.2f}')
plt.xlabel('Kilometers')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression Fit')
plt.savefig('try.png')
