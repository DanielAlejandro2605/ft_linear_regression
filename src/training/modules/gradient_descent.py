import numpy as np
import math
from typing import Union, Tuple

# Feature scaling
from modules.feature_scaling import standardization, denormalize_coefficients
# Cost function
from .cost_function import compute_cost_ft
# GIF functions from plotting
from .plotting import create_animation_frame, save_animation_gif

def partial_derivative_cost_function_of_w(data_x: np.ndarray, data_y: np.ndarray, w: float, b: float) -> float:
    m: int = data_x.shape[0]
    f_wb: np.ndarray = w * data_x + b
    deviation: np.ndarray = (f_wb - data_y) * data_x
    derivative_of_w: float = (1 / m) * np.sum(deviation)
    return derivative_of_w

def partial_derivative_cost_function_of_b(data_x: np.ndarray, data_y: np.ndarray, w: float, b: float) -> float:
    m: int = data_x.shape[0]
    f_wb: np.ndarray = w * data_x + b
    deviation: np.ndarray = f_wb - data_y
    derivative_of_b: float = (1 / m) * np.sum(deviation)
    return derivative_of_b

def gradient_descent(data_x: np.ndarray, data_y: np.ndarray, initial_w: float, initial_b: float, learning_rate: float, tolerance: float = 1e-8, max_iterations: int = 5000):
    w = initial_w
    b = initial_b
    frames = []
    cost_history = []
    
    for i in range(max_iterations):
        dj_dw = partial_derivative_cost_function_of_w(data_x, data_y, w, b)
        dj_db = partial_derivative_cost_function_of_b(data_x, data_y, w, b)
        
        new_w = w - learning_rate * dj_dw
        new_b = b - learning_rate * dj_db
        cost = compute_cost_ft(data_x, data_y, new_w, new_b)
        cost_history.append(cost)
        
        if i % 50 == 0:
            frame = create_animation_frame(data_x, data_y, new_w, new_b, cost_history, i)
            frames.append(frame)
        
        if abs(new_w - w) < tolerance and abs(new_b - b) < tolerance:
            print(f"Converged after {i} iterations.")
            break
        
        w = new_w
        b = new_b
    
    save_animation_gif(frames, '../../plots/gradient_descent_animation.gif')
    return w, b

def gradient_descent_no_animation(data_x: np.ndarray, data_y: np.ndarray, initial_w: float, initial_b: float, learning_rate: float, tolerance: float = 1e-8, max_iterations: int = 5000):
    w = initial_w
    b = initial_b
    
    for i in range(max_iterations):
        dj_dw = partial_derivative_cost_function_of_w(data_x, data_y, w, b)
        dj_db = partial_derivative_cost_function_of_b(data_x, data_y, w, b)
        
        new_w = w - learning_rate * dj_dw
        new_b = b - learning_rate * dj_db
        
        if i % 100 == 0:
            cost = compute_cost_ft(data_x, data_y, new_w, new_b)
            print(f"Iteration {i}: w={new_w:.6f}, b={new_b:.6f}, cost={cost:.6f}")
        
        if abs(new_w - w) < tolerance and abs(new_b - b) < tolerance:
            print(f"Converged after {i} iterations.")
            break
        
        w = new_w
        b = new_b
    
    return w, b

def save_coefficients_to_file(w_final: float, b_final: float, file_path: str) -> None:
    import json
    try:
        coefficients = {
            "w_final": w_final,
            "b_final": b_final
        }
        with open(file_path, 'w') as file:
            json.dump(coefficients, file, indent=2)
        print(f"Coefficients have been saved to {file_path}.")
    except IOError as e:
        print(f"An error occurred while trying to write to the file: {e}")

def lauch_gradient_descent(original_data_x: np.ndarray, original_data_y: np.ndarray, initial_w: float = 0, initial_b: float = 0) -> None:
    learning_rate: float = 0.01
    standardized_x: np.ndarray = standardization(original_data_x)
    
    w, b = gradient_descent(standardized_x, original_data_y, initial_w, initial_b, learning_rate)
    
    w_final, b_final = denormalize_coefficients(original_data_x, w, b)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    print(f"Animated GIF saved to plots/gradient_descent_animation.gif")
    
    save_coefficients_to_file(w_final, b_final, '../prediction/coefficients.json')

def lauch_gradient_descent_no_animation(original_data_x: np.ndarray, original_data_y: np.ndarray, initial_w: float = 0, initial_b: float = 0) -> None:
    learning_rate: float = 0.01
    standardized_x: np.ndarray = standardization(original_data_x)
    
    w, b = gradient_descent_no_animation(standardized_x, original_data_y, initial_w, initial_b, learning_rate)
    
    w_final, b_final = denormalize_coefficients(original_data_x, w, b)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    
    save_coefficients_to_file(w_final, b_final, '../prediction/coefficients.json')
