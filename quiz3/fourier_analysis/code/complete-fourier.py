import numpy as np
import matplotlib.pyplot as plt

def complete_fourier(t, A=1, alpha=0.5, R=1, L=0.5, omega=2*np.pi, N=1000):
    term1 = (A * alpha / R) * (1 - np.exp(-R * t / L))
    
    sum_term = np.zeros_like(t, dtype=complex)
    for k in range(-N, N + 1):
        if k == 0:
            continue  # Avoid division by zero
        
        numerator = A * np.sin(np.pi * k * alpha) * np.exp(-1j * np.pi * k * alpha)
        denominator = np.pi * k * (L * 1j * k * omega + R)
        factor = numerator / denominator
        sum_term += factor * (np.exp(1j * k * omega * t) - np.exp(-R * t / L))
    
    return term1 + sum_term.real

t = np.linspace(0, 10, 10000)
i_values = complete_fourier(t)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, i_values, label="$i(t)$", color='b')
plt.xlabel("Time (t)")
plt.ylabel("Current i(t)")
plt.title("Plot of i(t)")
plt.savefig("complete-fourier.png")
plt.legend()
plt.grid()
