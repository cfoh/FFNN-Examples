import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################
# Define the complex function
############################################################
def f(z):
    return z**2 + 2*z - 1j

############################################################
# Generate the dataset in csv
############################################################
def generate_csv(csv_filename='sample_data.csv', num_samples=5000):

    ## Generate random complex numbers within a specified range
    real_part = np.random.uniform(-2, 2, num_samples)
    imag_part = np.random.uniform(-2, 2, num_samples)

    ## Create an array of complex numbers
    z_samples = real_part + 1j * imag_part

    ## Evaluate the function at the random sample points
    w_samples = f(z_samples)

    ## Create a DataFrame with columns for real and imaginary parts
    data = {
        'Real(z)': np.real(z_samples),
        'Imag(z)': np.imag(z_samples),
        'Real(f(z))': np.real(w_samples),
        'Imag(f(z))': np.imag(w_samples)
    }

    df = pd.DataFrame(data)

    ## Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f'CSV file saved as {csv_filename}')

############################################################
# Plot the complex function
############################################################

def show_plot():

    # Create a grid of complex values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Evaluate the function at each point in the grid
    W = f(Z)

    # Create 3D plots for the real and imaginary parts
    fig = plt.figure(figsize=(12, 6))

    # Real part
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, np.real(W), cmap='viridis')
    ax1.set_xlabel('Real Axis')
    ax1.set_ylabel('Imaginary Axis')
    ax1.set_zlabel('Real(f(z))')
    ax1.set_title('Real Part of f(z)')

    # Imaginary part
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, np.imag(W), cmap='viridis')
    ax2.set_xlabel('Real Axis')
    ax2.set_ylabel('Imaginary Axis')
    ax2.set_zlabel('Imag(f(z))')
    ax2.set_title('Imaginary Part of f(z)')

    plt.tight_layout()
    plt.show()

############################################################
# Main
############################################################

generate_csv("sample_data.csv", num_samples=5000)
show_plot()
