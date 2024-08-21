import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cursor
import signal
import os
import time

from modules.signal_handler import signal_handler
from modules.plotting import plot_data,plot_with_regression_line, plot_deviation
from modules.get_regression_params import get_regression_params
signal.signal(signal.SIGINT, signal_handler)

def main_menu():
    """
    Displays the main menu for user interaction and processes user choices.
    """
    # Reading the file
    df = pd.read_csv('../data/data_example.csv')
    # Getting km data from dataframe
    data_km = df['km'].to_numpy()
    # Getting price data from dataframe
    data_price = df['price'].to_numpy()

    # # Get regression parameters from the user
    # w, b = get_regression_params()
    # Create the plot with the given parameters
    plot_deviation(data_km, data_price)

    # while True:
    #     os.system('clear')
    #     print("\n--- Main Menu ---")
    #     print("1. Plot raw data")
    #     print("2. Plot data with regression line")
    #     print("3. Exit")
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
    #             print("Exiting the program.")
    #             break
    #         case _:
    #             print("Invalid option. Please try again.")

    #     time.sleep(3)        


if __name__ == "__main__":
    main_menu()
