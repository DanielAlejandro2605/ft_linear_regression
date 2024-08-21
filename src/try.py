import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the file
df = pd.read_csv('../data/data_example.csv')

# Getting km data from dataframe
data_km = df['km'].to_numpy()
# Getting price data from dataframe
data_price = df['price'].to_numpy()

debug = True


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

# def compute_cost_ft(data_x, data_y, w=0.03, b=5000): 
#     # number of training examples
#     m = data_x.shape[0]

#     cost_sum = 0 
#     for i in range(m):
#         f_wb = w * data_x[i] + b
#         if debug:
#             print(f'f_xb = w * data_x[i] + b')
#             print(f'f_wb = {w } * {data_x[i]} + {b}')
#             print(f'f_wb = {f_wb}')
#         cost = (f_wb - data_y[i]) ** 2
#         if debug:
#             print(f'cost = (f_wb - data_y[i]) ** 2')
#             print(f'cost = ({f_wb} - {data_y[i]}) ** {2}')
#             print(f'cost = {cost}')
#         cost_sum = cost_sum + cost
#         if debug:
#             print(cost_sum)
#             print('-------------')
#     total_cost = (1 / (2 * m)) * cost_sum  

#     return total_cost