import numpy as np
import matplotlib.pyplot as plt

def compute_current(t, A, alpha, T, R, L, N):
    omega_0 = 2 * np.pi / T  
    I_dc = (A * alpha) / (R)  
    
    i_t = np.full_like(t, I_dc)  
    
    for n in range(1, N + 1):  
        sin_term = np.sin(n * np.pi * alpha)
        magnitude = A * sin_term / (n * np.pi)
        attenuation_R = (2 * R)/ (R**2 + (n * omega_0 * L)**2)
        attenuation_L = (2 * n * omega_0 * L) / (R**2 + (n * omega_0 * L)**2)
        phase_shift = n * np.pi * alpha
        
        cos_component = magnitude * attenuation_R * np.cos(n * omega_0 * t - phase_shift)
        sin_component = magnitude * attenuation_L * np.sin(n * omega_0 * t - phase_shift)
        
        i_t += cos_component + sin_component
    
    return i_t

def root_mean_square_percentage_error(N_1, N_2, t, A, alpha, T, R, L):
    current_1 = compute_current(t, A, alpha, T, R, L, N_2)
    current_2 = compute_current(t, A, alpha, T, R, L, N_1)
    
    mask = current_1 != 0 
    errors = np.zeros_like(current_1)
    errors[mask] = ((current_1[mask] - current_2[mask]) / current_1[mask]) ** 2 
    root_mean_square_percentage_error_value = np.sqrt(np.mean(errors)) * 100

    return root_mean_square_percentage_error_value
    

A = 5          # Amplitude of square wave
alpha = 0.3    # Duty cycle (0 < alpha < 1)
T = 1e-3       # Period of the square wave (1ms)
R = 10         # Resistance in ohms
L = 1e-2       # Inductance in Henrys
N = 20000      # Number of harmonics

t = np.linspace(0, 2*T, 1000)  
current_0 = compute_current(t, A, 0.1, T, R, L, N)
current_1 = compute_current(t, A, 0.2, T, R, L, N)
current_2 = compute_current(t, A, 0.3, T, R, L, N)
current_3 = compute_current(t, A, 0.4, T, R, L, N)
current_4 = compute_current(t, A, 0.5, T, R, L, N)
current_5 = compute_current(t, A, 0.6, T, R, L, N)
current_6 = compute_current(t, A, 0.7, T, R, L, N)
current_7 = compute_current(t, A, 0.8, T, R, L, N)
current_8 = compute_current(t, A, 0.9, T, R, L, N)

plt.plot(t, current_0, label="alpha = 0.1")
plt.plot(t, current_1, label="alpha = 0.2")
plt.plot(t, current_2, label="alpha = 0.3")
plt.plot(t, current_3, label="alpha = 0.4")
plt.plot(t, current_4, label="alpha = 0.5")
plt.plot(t, current_5, label="alpha = 0.6")
plt.plot(t, current_6, label="alpha = 0.7")
plt.plot(t, current_7, label="alpha = 0.8")
plt.plot(t, current_8, label="alpha = 0.9")
# Plot
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("alpha_variation.png")
plt.show()
