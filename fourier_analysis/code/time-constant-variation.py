import numpy as np
import matplotlib.pyplot as plt

def complete_fourier(t, A=1, alpha=0.5, R=1, L=0.5, omega=2*np.pi, N=1000):
    term1 = (A * alpha / R) * (1 - np.exp(-R * t / L))
    
    sum_term = np.zeros_like(t, dtype=complex)
    for k in range(-N, N + 1):
        if k == 0:
            continue  
        
        numerator = A * np.sin(np.pi * k * alpha) * np.exp(-1j * np.pi * k * alpha)
        denominator = np.pi * k * (L * 1j * k * omega + R)
        factor = numerator / denominator
        sum_term += factor * (np.exp(1j * k * omega * t) - np.exp(-R * t / L))
    
    return term1 + sum_term.real

t = np.linspace(0, 10, 10000)
omega = 2 * np.pi 
T = 2 * np.pi / omega 

L_R_values = [50 * T, T, 0.02 * T]
labels = [r"$L/R \gg T$", r"$L/R = T$", r"$L/R \ll T$"]
colors = ['b', 'g', 'r']

plt.figure(figsize=(10, 5))

for L_R, label, color in zip(L_R_values, labels, colors):
    L = L_R  
    R = 1    
    i_values = complete_fourier(t, R=R, L=L, omega=omega)
    plt.plot(t, i_values, label=label, color=color)

plt.xlabel("Time (t)")
plt.ylabel("Current i(t)")
plt.title("Effect of L/R Ratio on i(t)")
plt.legend()
plt.grid()
plt.savefig("../figs/time-constant.png")
plt.show()

