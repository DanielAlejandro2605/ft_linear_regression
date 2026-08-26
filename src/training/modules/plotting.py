import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from .cost_function import compute_cost_ft

def plot_data(data_km: np.ndarray, data_price: np.ndarray, plot_path: str = '../../plots/plot_data.png') -> None:
    """
    Plots the data points as a scatter plot.

    Args:
        data_km (np.ndarray): Feature data representing kilometers.
        data_price (np.ndarray): Target values representing prices.
        plot_path (str, optional): Path to save the plot image.

    Returns:
        None
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()

    # Create a scatter plot of the data
    plt.scatter(data_km, data_price, color='blue', marker='x')

    # Set labels and title
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Repartition of data')

    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')


def plot_with_regression_line(data_km: np.ndarray, data_price: np.ndarray, w: float, b: float, plot_path: str = '../../plots/plot_regression_line.png') -> None:
    """
    Plots the data points and a regression line based on the given parameters.

    Args:
        data_km (np.ndarray): Feature data representing kilometers.
        data_price (np.ndarray): Target values representing prices.
        w (float): Slope of the regression line.
        b (float): Y-intercept of the regression line.
        plot_path (str, optional): Path to save the plot image.

    Returns:
        None
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
    plt.plot(x_range, y_range, color='red', label=f'Regression: y = {w:.4f}x + {b:.4f}')
    
    # Set labels and title
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')


def plot_deviation(data_km: np.ndarray, data_price: np.ndarray, w: float = 0.03, b: float = 5000, plot_path: str = '../../plots/plot_deviation.png') -> None:
    """
    Displays the deviation of each data point from the regression line.

    Args:
        data_km (np.ndarray): Feature data representing kilometers.
        data_price (np.ndarray): Target values representing prices.
        w (float, optional): Slope of the regression line.
        b (float, optional): Y-intercept of the regression line.
        plot_path (str, optional): Path to save the plot image.

    Returns:
        None
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

def plot_cost_function_only_w(data_km: np.ndarray, data_price: np.ndarray, plot_path: str = '../../plots/plot_cost_function_only_w.png') -> None:
    """
    Plots the cost function J(w) while varying only the slope (w) and keeping the intercept (b) fixed.

    Args:
        data_km (np.ndarray): Feature data
        data_price (np.ndarray): Target values
        plot_path (str, optional): The path where the plot will be saved.

    Returns:
        None
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()
    
    # Define ranges for w
    w_values = np.linspace(-2.5, 2.5, 100)

    # Compute the cost for each value of w while b is fixed
    j_w = np.array([compute_cost_ft(data_km, data_price, w) for w in w_values])

    # Plot the cost function J(w)
    plt.plot(w_values, j_w)
    
    # Setting labels and title
    plt.xlabel('w')
    plt.ylabel('J(w)')
    plt.title('Cost Function J(w)')
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')


def plot_cost_function_only_b(data_km: np.ndarray, data_price: np.ndarray, plot_path: str = '../../plots/plot_cost_function_only_b.png') -> None:
    """
    Plots the cost function J(b) while varying only the intercept (b) and keeping the slope (w) fixed.

    Args:
        data_km (np.ndarray): Feature data.
        data_price (np.ndarray): Target values.
        plot_path (str, optional): The path where the plot will be saved.

    Returns:
        None
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()
    
    # Define ranges for b
    b_values = np.linspace(0, 10000, 100)

    # Compute the cost for each value of b while w is fixed
    j_b = np.array([compute_cost_ft(data_km, data_price, 0, b) for b in b_values])

    # Plot the cost function J(b)
    plt.plot(b_values, j_b)
    
    # Setting labels and title
    plt.xlabel('b')
    plt.ylabel('J(b)')
    plt.title('Cost Function J(b)')
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')

def plot_cost_function_scatter(iterations: list, costs: list, plot_path: str = '../../plots/cost_function_scatter.png') -> None:
    """
    Creates a scatter plot of the cost function versus iterations.

    Args:
        iterations (list): List of iteration numbers.
        costs (list): List of corresponding cost values.
        plot_path (str, optional): Path where the plot will be saved.

    Returns:
        None
    """
    # Clear the current figure to prevent overlaying of plots
    plt.clf()
    
    # Create scatter plot
    plt.scatter(iterations, costs, color='blue', marker='o')
    
    # Setting labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Scatter Plot')
    
    # Save the plot
    plt.savefig(plot_path)
    print(f'The plot has been saved in {plot_path}!')

def create_animation_frame(data_x: np.ndarray, data_y: np.ndarray, w: float, b: float, cost_history: list, iteration: int) -> Image.Image:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(data_x, data_y, alpha=0.6, color='blue', label='Data')
    x_line = np.linspace(data_x.min(), data_x.max(), 100)
    y_line = w * x_line + b
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression Line (w={w:.4f}, b={b:.4f})')
    ax1.set_xlabel('Kilometers')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Gradient Descent - Iteration {iteration}')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(range(len(cost_history)), cost_history, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost Function')
    ax2.grid(True)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    frame = Image.open(buf)
    plt.close()
    
    return frame

def save_animation_gif(frames: list, output_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if frames:
        frames[0].save(output_path, 
                      save_all=True, 
                      append_images=frames[1:], 
                      duration=200, 
                      loop=0)
        print(f"GIF animation saved to {output_path}")
    else:
        print("No frames created!")
