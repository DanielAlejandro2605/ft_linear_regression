import numpy as np
import math

# Cost function
from .cost_function import compute_cost_ft

# Derivatives (partial derivatives) are numbers (scalars) that indicate the slope of the cost function at a specific point.
# If the derivative is positive, the cost function is increasing, which means we need to decrease the parameter value to decrease the cost.
# If the derivative is negative, the cost function is decreasing, which means we need to increase the parameter value.
# If the derivative is close to zero, it means we are close to a minimum point.

def partial_derivative_cost_function_of_w(data_x, data_y, w, b):
    # Number of training examples
    m = data_x.shape[0]
    # Compute linear function hypothesis
    f_wb = w * data_x + b

    # Compute the deviation from the real value scaled by x
    deviation = (f_wb - data_y) * data_x
    # Getting the derivative (sum of deviation)
    derivative_of_w = (1 / m) * np.sum(deviation)
    return derivative_of_w
    
def partial_derivative_cost_function_of_b(data_x, data_y, w, b):
    # Number of training examples
    m = data_x.shape[0]
    # Compute linear function hypothesis
    f_wb = w * data_x + b
    # Compute the deviation from the real value
    deviation = f_wb - data_y
    # Getting the derivative (sum of deviation)
    derivative_of_b = (1 / m) * np.sum(deviation)
    return derivative_of_b

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i


    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(data_x, data_y, initial_w=0, initial_b=0):
    # Start with some w, b, as zero for example
    print("Gradient descent!")

    # Learning rate (always a positive number) controls how big will be the step in the gradient descent loop until search the minimun
    # If the learning_rate is too small -> Gradient descent may be slow
    # If the learning rate is too large -> Gradient descent may overshoot, so never reach minimun
    learning_rate = 0.01

    # Derivative defines the direction of the step

    # Keep changing w,b to reduce J(w,b) cost function until we settle at or near a minimun
    # Repeat until convergence
    # Convergence is the property of two or more things ending at the same point.

    # Near a local minimun:
    # - Derivative becomes smaller
    # - Update steps become smaller
    # Make a condition for this? 
    w = initial_w
    b = initial_b

    for i in range(10000):
        dj_dw, dj_db = compute_gradient(data_x, data_y, w , b)

        if math.isnan(w) or math.isnan(b) or math.isinf(w) or math.isinf(b):
            print("Overflow detected! Stopping gradient descent.")
            break


        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

    return w,b
