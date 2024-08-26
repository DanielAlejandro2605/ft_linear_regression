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
    df = pd.read_csv('../data/data_example.csv')

    data_frame = df.to_numpy()

    # Getting km data from dataframe
    original_data_km = df['km'].to_numpy()
    # Getting price data from dataframe
    original_data_price = df['price'].to_numpy()
    

    # partial_derivative_of_w(data_km, data_price)
    # partial_derivative_of_b(data_km, data_price)



    # Calculate the Frobenius norm
    frobenius_norm = np.linalg.norm(data_frame, 'fro')

    # Normalize the matrix
    normalized_matrix = data_frame / frobenius_norm

    # print('Original Matrix:')
    # print(data_frame)
    # print('nFrobenius Norm:')
    # print(frobenius_norm)
    # print('nNormalized Matrix:')
    # print(normalized_matrix)

    print(normalized_matrix[0])

    dataset = pd.DataFrame({'km': normalized_matrix[:, 0], 'price': normalized_matrix[:, 1]})

    print(dataset)
    # Getting km data from dataframe
    data_km = dataset['km'].to_numpy()
    # Getting price data from dataframe
    data_price = dataset['price'].to_numpy()

    w_final, b_final = gradient_descent(data_km, data_price)

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
