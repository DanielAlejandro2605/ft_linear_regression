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
from modules.plotting import plot_data
# Gradient descent
from modules.gradient_descent import lauch_gradient_descent, lauch_gradient_descent_no_animation

# Setting signal
signal.signal(signal.SIGINT, signal_handler)

def main_menu():
    """
    Displays the main menu for user interaction and processes user choices.
    """
    try:
        # Reading the file
        df = pd.read_csv('../../data/data.csv')
        # Getting km data from dataframe
        original_data_km = df['km'].to_numpy()
        # Getting price data from dataframe
        original_data_price = df['price'].to_numpy()
    except FileNotFoundError:
        print("Error: data.csv file not found!")
        print("Please make sure the data file exists at: data/data.csv")
        print("Exiting...")
        return
    except Exception as e:
        print(f"Error reading data file: {e}")
        print("Exiting...")
        return
        
    actions = {
        '1': lambda: plot_data(original_data_km, original_data_price),
        '2': lambda: lauch_gradient_descent_no_animation(original_data_km, original_data_price),
        '3': lambda: lauch_gradient_descent(original_data_km, original_data_price),
        '4': exit_program,
    }

    while True:
        print("\n--- Main Menu ---")
        print("1. Plot data")
        print("2. Run gradient descent algorithm (no animation)")
        print("3. Run gradient descent algorithm (creates animated GIF)")
        print("4. Exit")
        choice = input("Choose an option: ")

        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid option. Please try again.")
        
        time.sleep(1)

def exit_program():
    print("Exiting the program.")
    exit()

if __name__ == "__main__":
    main_menu()