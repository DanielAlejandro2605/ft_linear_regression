import numpy as np

def compute_cost_ft(data_x: np.ndarray, data_y: np.ndarray, w: float = 0.03, b: float = 5000) -> float:
    """
    Computes the cost function (Squared Error Cost Function) for linear regression.

    Args:
        data_x (np.ndarray): Feature data
        data_y (np.ndarray): Target values
        w (float, optional): Slope of the regression line. Default is 0.03.
        b (float, optional): Y-intercept of the regression line. Default is 5000.

    Returns:
        float: The cost of using `w` and `b` as parameters for linear regression to fit 
               the data points in `data_x` and `data_y`.
    """
    # Number of training examples
    m: int = data_x.shape[0]

    # Computation of the predictions: f_wb = w * data_x + b
    f_wb: np.ndarray = w * data_x + b
    
    # Compute the squared errors
    errors: np.ndarray = (f_wb - data_y) ** 2
    
    # Compute the cost (mean squared error)
    total_cost: float = (1 / (2 * m)) * np.sum(errors)
    
    return total_cost
