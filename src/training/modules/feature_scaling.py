import numpy as np
from typing import Union
from typing import Tuple

def standardization(data: np.ndarray) -> np.ndarray:
    """
    Applies Z-score standardization to the input data.

    Z-score standardization transforms the data such that it has a mean of 0 and a standard deviation of 1.

    Args:
        data (np.ndarray): The input data to be standardized.

    Returns:
        np.ndarray: The standardized data.
    """

    mean_data: float = np.mean(data)
    standard_deviation_data: float = np.std(data)
    standardized_data: np.ndarray = (data - mean_data) / standard_deviation_data
    return standardized_data

def denormalize_coefficients(data: np.ndarray, w_normalized: float, b_normalized: float) -> Tuple[float, float]:
    """
    Denormalizes the coefficients after applying Z-score standardization.

    Given that the data was standardized using Z-score, this function returns the original 
    slope (w) and intercept (b) for the regression model in the original scale of the data.

    Args:
        data (np.ndarray): The original feature data used for standardization.
        w_normalized (float): The slope coefficient obtained after fitting the model on standardized data.
        b_normalized (float): The intercept obtained after fitting the model on standardized data.

    Returns:
        Tuple[float, float]: A tuple containing the denormalized slope and intercept (w_original, b_original).
    """
    mean_data: float = np.mean(data)
    standard_deviation_data: float = np.std(data)

    w_original: float = w_normalized / standard_deviation_data
    b_original: float = b_normalized - (w_normalized * mean_data / standard_deviation_data)
    return w_original, b_original