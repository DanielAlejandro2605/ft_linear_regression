import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cursor
import signal
import os
import time

# Signal
from modules.signal_handler import signal_handler
# Plotting
from modules.plotting import plot_data,plot_with_regression_line, \
                            plot_deviation, \
                            plot_cost_function_only_w, \
                            plot_cost_function_only_b
# Feature scaling
from modules.feature_scaling import normalization, standardization, denormalize_coefficients, denormalize_coefficients_2
# Gradient descent
from modules.gradient_descent_try import gradient_descent
# Get params
from modules.get_regression_params import get_regression_params

# Setting signal
signal.signal(signal.SIGINT, signal_handler)

def main_menu():
    """
    Displays the main menu for user interaction and processes user choices.
    """
    # Reading the file
    df = pd.read_csv('../data/data.csv')
    # Transform dataframe to numpy array
    data_frame = df.to_numpy()
    # Getting km data from dataframe
    original_data_km = df['km'].to_numpy()
    # Getting price data from dataframe
    original_data_price = df['price'].to_numpy()

    # Normalized data
    normalized_x = normalization(original_data_km)
    # print(normalized_x)
    print('---------------------------------------')
    # Standardized data
    standardized_x = standardization(original_data_km)
    # print(standardized_x)
    # Gradient descent    

    # partial_derivative_of_w(data_km, data_price)
    # partial_derivative_of_b(data_km, data_price)

    w, b = gradient_descent(standardized_x, original_data_price)

    print(f'{w, b}')

    w_final, b_final = denormalize_coefficients_2(original_data_km, w, b)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    plot_with_regression_line(original_data_km, original_data_price, w_final, b_final)

    # while True:
    #     os.system('clear')
    #     print("\n--- Main Menu ---")
    #     print("1. Plot raw data")
    #     print("2. Plot data with regression line for hypothesis")
    #     print("3. Plot data with regression line for hypothesis and deviation")
    #     print("4. Plot cost function only with 'w' parameter")
    #     print("5. Plot cost function only with 'b' parameter")
    #     print("6. Exit")
    #     choice = input("Choose an option: ")

    #     match choice:
    #         case '1':
    #             # Plotting data
    #             plot_data(data_km, data_price)
    #         case '2':        
    #             # Get regression parameters from the user
    #             w, b = get_regression_params()
    #             # Create the plot with the given parameters
    #             plot_with_regression_line(data_km, data_price, w, b)
    #         case '3':
    #             # Get regression parameters from the user
    #             w, b = get_regression_params()
    #             # Create the plot with the given parameters and show deviation
    #             plot_deviation(data_km, data_price, w, b)
    #         case '4':
    #             plot_cost_function_only_w(data_km, data_price)
    #         case '5':
    #             plot_cost_function_only_b(data_km, data_price)
    #         case '6':
    #             print("Exiting the program.")
    #             break
    #         case _:
    #             print("Invalid option. Please try again.")

    #     time.sleep(3)        


if __name__ == "__main__":
    main_menu()
