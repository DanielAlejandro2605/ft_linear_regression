from typing import Tuple

def get_regression_params() -> Tuple[float, float]:
    """
    Prompts the user to input values for the slope (w) and y-intercept (b) of the regression line.

    Ensures that:
    - The slope (w) is within the range of -0.03 to 0.03.
    - The intercept (b) is within the range of 0 to 10,000.

    Returns:
        Tuple[float, float]: A tuple containing the values for w (slope) and b (intercept).
    """
    while True:
        try:
            w: float = float(input("Enter the value of w (slope) [-0.03 to 0.03]: "))
            if not (-0.03 <= w <= 0.03):
                print("The slope (w) should be between -0.03 and 0.03. Please try again.")
                continue

            b: float = float(input("Enter the value of b (intercept) [0 <= b <= 10000]: "))
            if not (0 <= b <= 10000):
                print("The intercept (b) should be between 0 and 10,000. Please try again.")
                continue

            break

        except ValueError:
            print("Invalid input. Please enter numerical values.")
    
    return w, b
