import numpy as np

def normalization(data):
    min_value, max_value = np.min(data), np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

def standardization(data):
    mean_data = np.mean(data)
    standard_deviation_data = np.std(data)
    standardized_data = (data - mean_data) / standard_deviation_data
    return standardized_data


# For min-max
def denormalize_coefficients(data_x, data_y, w_normalized, b_normalized):
    x_min, x_max = np.min(data_x), np.max(data_x)
    y_min, y_max = np.min(data_y), np.max(data_y)
    
    w_original = w_normalized * (x_max - x_min) / (y_max - y_min)
    
    b_original = y_min - (w_original * x_min) + b_normalized * (y_max - y_min)
    
    return w_original, b_original

# For Z-Score
def denormalize_coefficients_2(data, w_normalized, b_normalized):
    
    mean_data = np.mean(data)
    standard_deviation_data = np.std(data)

    w_original = w_normalized / standard_deviation_data
    b_original = b_normalized - (w_normalized * mean_data / standard_deviation_data)

    return w_original, b_original