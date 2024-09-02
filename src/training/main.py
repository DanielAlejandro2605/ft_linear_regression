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

# Gradient descent
from modules.gradient_descent import lauch_gradient_descent
# Get params
from modules.get_regression_params import get_regression_params
# Cost function
from modules.cost_function import compute_cost_ft

# Setting signal
signal.signal(signal.SIGINT, signal_handler)

def main_menu():
    """
    Displays the main menu for user interaction and processes user choices.
    """
    # Reading the file
    df = pd.read_csv('../../data/data.csv')
    # Transform dataframe to numpy array
    data_frame = df.to_numpy()
    # Getting km data from dataframe
    original_data_km = df['km'].to_numpy()
    # Getting price data from dataframe
    original_data_price = df['price'].to_numpy()

    lauch_gradient_descent(original_data_km, original_data_price)
    
    actions = {
        '1': lambda: plot_data(original_data_km, original_data_price),
        '2': lambda: plot_with_regression_line(original_data_km, original_data_price, *get_regression_params()),
        '3': lambda: plot_deviation(original_data_km, original_data_price, *get_regression_params()),
        '4': lambda: plot_cost_function_only_w(original_data_km, original_data_price),
        '5': lambda: plot_cost_function_only_b(original_data_km, original_data_price),
        '6': lambda: lauch_gradient_descent(original_data_km, original_data_price),
        '7': exit_program
    }

    # while True:
    #     print("\n--- Main Menu ---")
    #     print("1. Plot raw data")
    #     print("2. Plot data with regression line for hypothesis")
    #     print("3. Plot data with regression line for hypothesis and deviation")
    #     print("4. Plot cost function only with 'w' parameter")
    #     print("5. Plot cost function only with 'b' parameter")
    #     print("6. Lauch gradient descent algorithm")
    #     print("7. Exit")
    #     choice = input("Choose an option: ")

    #     action = actions.get(choice)
    #     if action:
    #         action()
    #     else:
    #         print("Invalid option. Please try again.")
        
    #     time.sleep(3)

def exit_program():
    print("Exiting the program.")
    exit()

if __name__ == "__main__":
    main_menu()