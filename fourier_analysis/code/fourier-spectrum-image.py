import numpy as np
import matplotlib.pyplot as plt

def plot_fourier_coefficients(ax, A, alpha, num_terms):
    k_values = np.arange(-num_terms, num_terms + 1)
    k_values = k_values[k_values != 0]  # Exclude k = 0
    
    coefficients = (A / (np.pi * k_values)) * np.sin(np.pi * k_values * alpha) * np.exp(-1j * np.pi * k_values * alpha)
    
    ax.stem(k_values, np.abs(coefficients))
    ax.set_xlabel('k')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Alpha = {alpha}')
    ax.grid()

A = 1.0
num_terms = 50
alphas = [0.035, 0.5, 0.0885, 0.973]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Fourier Coefficients for Different Alpha Values')

for ax, alpha in zip(axes.flat, alphas):
    plot_fourier_coefficients(ax, A, alpha, num_terms)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("../figs/fourier-spectrum.png")
plt.show()

