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
        file_path (str): Path to the file where the coefficients are stored.

    Returns:
        Tuple[float, float]: A tuple containing the slope (w) and intercept (b) of the regression line.
    
    Raises:
        ValueError: If the file does not contain valid float values or has an unexpected format.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) != 2:
                raise ValueError("The file does not contain the expected number of lines.")

            w_line = lines[0].strip()
            b_line = lines[1].strip()

            # Extracting float values from the lines
            w_final = float(w_line.split(':')[1].strip())
            b_final = float(b_line.split(':')[1].strip())

        return w_final, b_final
    
    except (IOError, ValueError) as e:
        print(f"An error occurred while trying to read the file: {e}")
        raise

file_path = 'coefficients.txt'

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
    # Making prediction
    estimate_price(w_final, b_final)
except ValueError as e:
    print(f"Failed to load coefficients: {e}")
