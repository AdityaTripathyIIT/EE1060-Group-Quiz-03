import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_fourier_coefficients(A, alpha, num_terms):
    k_values = np.arange(-num_terms, num_terms + 1)
    k_values = k_values[k_values != 0]  # Exclude k = 0
    
    coefficients = (A / (np.pi * k_values)) * np.sin(np.pi * k_values * alpha) * np.exp(-1j * np.pi * k_values * alpha)
    
    ax.clear()
    ax.stem(k_values, np.abs(coefficients))  # Removed use_line_collection for compatibility
    ax.set_xlabel('k')
    ax.set_ylabel('Magnitude')
    ax.set_title('Fourier Coefficients Spectrum')
    ax.grid()
    fig.canvas.draw_idle()

def update(val):
    A = slider_A.val
    alpha = slider_alpha.val
    num_terms = int(slider_num_terms.val)
    plot_fourier_coefficients(A, alpha, num_terms)

# Initial parameters
A_init = 1.0
alpha_init = 0.5
num_terms_init = 50

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

# Sliders
ax_A = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_alpha = plt.axes([0.15, 0.06, 0.65, 0.03])
ax_num_terms = plt.axes([0.15, 0.02, 0.65, 0.03])

slider_A = Slider(ax_A, 'A', 0.1, 5.0, valinit=A_init)
slider_alpha = Slider(ax_alpha, 'alpha', 0.0, 1.0, valinit=alpha_init)
slider_num_terms = Slider(ax_num_terms, 'num_terms', 10, 100, valinit=num_terms_init, valstep=1)

slider_A.on_changed(update)
slider_alpha.on_changed(update)
slider_num_terms.on_changed(update)

# Initial plot
plot_fourier_coefficients(A_init, alpha_init, num_terms_init)
plt.show()

