import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reading the file
df = pd.read_csv('../../data/data_example.csv')

# Getting km data from dataframe
data_km = df['km'].to_numpy()
data_price = df['price'].to_numpy()

def compute_cost_ft(data_x, data_y, w=0.03, b=5000): 
    """
    Computes the cost function for linear regression using vectorization.
    
    Args:
      data_x (ndarray): Feature data (e.g., kilometers).
      data_y (ndarray): Target values (e.g., prices).
      w (float): Slope of the regression line.
      b (float): Y-intercept of the regression line.
    
    Returns:
        total_cost (float): The cost of using w, b as the parameters for linear regression
                            to fit the data points in data_x and data_y.
    """
    # Number of training examples
    m = data_x.shape[0]

    # Vectorized computation of the predictions: f_wb = w * data_x + b
    f_wb = w * data_x + b
    
    # Compute the squared errors
    errors = (f_wb - data_y) ** 2
    
    # Compute the cost (mean squared error)
    total_cost = (1 / (2 * m)) * np.sum(errors)
    
    return total_cost

# Define ranges for w and b
w_values = np.linspace(-0.03, 0.03, 100)
# print(w_values)
b_values = np.linspace(0, 10000, 100)

# Create a grid of w and b values
W, B = np.meshgrid(w_values, b_values)

print(W)
print(B)

# Compute the cost for each combination of w and b
Z = np.array([compute_cost_ft(data_km, data_price, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

# Plotting the cost function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Z, cmap='viridis')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost')
ax.set_title('Cost Function J(w, b)')

plt.savefig('try.png')
