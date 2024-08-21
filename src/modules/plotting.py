import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data_km, data_price, plot_path='../plots/plot_data.png'):
    """
    Plots the data points.

    Args:
    data_km : array-like
        Kilometers data.
    data_price : array-like
        Price data.
    plot_path (str)
        Path to save the plot image.
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()

    # To show the relationship between two numerical variables
    plt.scatter(data_km, data_price, color='blue', marker='x')

    # Labels
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Scatter Plot')

    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')

def plot_with_regression_line(data_km, data_price, w, b, plot_path='../plots/plot_regression_line.png'):
    """
    Plots the data points and a regression line based on the given parameters.
    
    Args:
    data_km : array-like
        Kilometers data.
    data_price : array-like
        Price data.
    w (float)
        Slope of the regression line.
    b (float)
        Y-intercept of the regression line.
    plot_path (str)
        Path to save the plot image.
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()

    # Create a scatter plot of the data
    plt.scatter(data_km, data_price, color='blue', marker='x', label='Data')
    
    # Define the range of x values for plotting the regression line
    x_range = np.linspace(min(data_km), max(data_km), 100)
    
    # Compute y values for the regression line
    y_range = w * x_range + b
    
    # Plot the regression line
    plt.plot(x_range, y_range, color='red', label=f'Regression: y = {w}x + {b}')
    
    # Set labels and title
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')


def plot_deviation(data_km, data_price, w=0.03, b=5000, plot_path='../plots/plot_deviation.png'):
    """
    Displays the deviation of each data point with respect to the regression line.
    
    Args:
    data_km : array-like
        Kilometers data.
    data_price : array-like
        Price data.
    w (float)
        Slope of the regression line.
    b (float)
        Y-intercept of the regression line.
    plot_path (str)
        Path to save the plot image.
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()
    
    # Create a scatter plot of the data
    plt.scatter(data_km, data_price, color='blue', marker='x', label='Data')
    
    # Define the range of x values for plotting the regression line
    x_range = np.linspace(min(data_km), max(data_km), 100)
    
    # Compute y values for the regression line
    y_range = w * x_range + b
    
    # Plot the regression line
    plt.plot(x_range, y_range, color='red', label=f'Regression: y = {w}x + {b}')
    
    # Calculate predicted prices for each km
    predicted_prices = w * data_km + b

    print(len(predicted_prices))
    print(predicted_prices)
    print(predicted_prices[0])
    print(predicted_prices[1])
    
    # Draw vertical lines from each data point to the regression line
    for km, observed_price, predicted_price in zip(data_km, data_price, predicted_prices):
        plt.plot([km, km], [observed_price, predicted_price], color='gray', linestyle='--', linewidth=0.7)
    
    # Set labels and title
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Scatter Plot with Regression Line and Deviations')
    plt.legend()
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')
