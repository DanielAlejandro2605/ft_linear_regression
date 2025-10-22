from typing import Tuple
import sys
import signal
import time

def signal_handler(sig, frame):
    print("\nYou have pressed CTRL+C")
    print("Goodbye. See you around!")
    time.sleep(2)
    sys.exit(0)


def load_coefficients_from_file(file_path: str) -> Tuple[float, float]:
    """
    Loads the final coefficients of the regression line from a specified file.

    Args:
        file_path (str): Path to the JSON file where the coefficients are stored.

    Returns:
        Tuple[float, float]: A tuple containing the slope (w) and intercept (b) of the regression line.
        Returns (0.0, 0.0) if file doesn't exist or is invalid.
    """
    import json
    import os
    
    if not os.path.exists(file_path):
        print(f"Warning: Coefficients file '{file_path}' not found.")
        print("Using default values: w_final = 0.0, b_final = 0.0")
        print("Please run training first to generate proper coefficients.")
        return 0.0, 0.0
    
    try:
        with open(file_path, 'r') as file:
            coefficients = json.load(file)
            
            if 'w_final' not in coefficients or 'b_final' not in coefficients:
                print("Warning: JSON file does not contain the expected coefficient keys.")
                print("Using default values: w_final = 0.0, b_final = 0.0")
                return 0.0, 0.0

            w_final = float(coefficients['w_final'])
            b_final = float(coefficients['b_final'])

        return w_final, b_final
    
    except (IOError, ValueError, json.JSONDecodeError) as e:
        print(f"Error reading coefficients file: {e}")
        print("Using default values: w_final = 0.0, b_final = 0.0")
        return 0.0, 0.0

file_path = 'coefficients.json'

def estimate_price(w_final: float, b_final: float):

    kms_to_predict : float = 0.0
    while True:
        try:
            kms_to_predict: float = float(input("Enter the value of kms to predict: "))
            if not (kms_to_predict > 0):
                print("The value of kilometers must be positive.Please try again.")
                continue
            break

        except ValueError:
            print("Invalid input. Please enter numerical values.")

    price : float = w_final * kms_to_predict + b_final

    print(f"A car with {kms_to_predict} has a price of {price:.4f}")

try:
    # Setting signal
    signal.signal(signal.SIGINT, signal_handler)
    # Getting coefficients from file
    w_final, b_final = load_coefficients_from_file(file_path)
    print(f"Loaded coefficients: w_final = {w_final:.4f}, b_final = {b_final:.4f}")
    
    # Check if coefficients are default values (indicating no training has been done)
    if w_final == 0.0 and b_final == 0.0:
        print("\n" + "="*50)
        print("WARNING: Using default coefficients (0.0, 0.0)")
        print("="*50 + "\n")
    
    # Making prediction
    estimate_price(w_final, b_final)
except Exception as e:
    print(f"Failed to load coefficients: {e}")
