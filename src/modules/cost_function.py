import numpy as np

def compute_cost_ft(data_x, data_y, w=0.03, b=5000): 
    """
    Computes the cost function (Squared error cost function) for linear regression
    
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

    # Computation of the predictions: f_wb = w * data_x + b
    f_wb = w * data_x + b
    
    # Compute the squared errors
    errors = (f_wb - data_y) ** 2
    
    # Compute the cost (mean squared error)
    total_cost = (1 / (2 * m)) * np.sum(errors)
    
    return total_cost