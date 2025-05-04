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
N = 2000         # Number of harmonics

steady_current = (A/R) * (np.exp(-R*T/L)) * (np.exp(R * alpha * T/L) - 1)/(1 - np.exp(-R*T/L))
peak = A/R + (steady_current - A/R)*np.exp(-R*alpha*T/L)

t = np.linspace(0, 2*T, 1000)  
plt.scatter(0.3, peak)
plt.plot(t * 1e3, compute_current(t, A, alpha, T, R, L, 1000), label=f'N = {N} harmonics')
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
plt.title("RL Circuit Response to Square Wave Input")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("steady-state.png")
plt.show()
