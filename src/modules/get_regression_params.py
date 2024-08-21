def get_regression_params():
    """
    Prompts the user to input values for the slope (w) and y-intercept (b) of the regression line.
    
    Returns:
    tuple: A tuple containing the values for w and b.
    """
    while True:
        try:
            # Prompt user for the slope (w)
            w = float(input("Enter the value of w (slope) [-0.03 to 0.03]: "))
            
            # Basic check for the slope, adjust the range based on typical values
            if not (-0.03 <= w <= 0.03):
                print("The slope (w) should be between -10 and 10. Please try again.")
                continue
            
            # Prompt user for the y-intercept (b)
            b = float(input("Enter the value of b (intercept) [0 <= b <= 10000]: "))
            
            # Basic check for the intercept, adjust the range based on typical price values
            if not (0 <= b <= 10000):
                print("The intercept (b) should be between 0 and 10,000. Please try again.")
                continue
            
            # If both inputs are valid, break the loop
            break
            
        except ValueError:
            print("Invalid input. Please enter numerical values.")
    
    return w, b
