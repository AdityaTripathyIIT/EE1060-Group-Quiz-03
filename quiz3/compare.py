import numpy as np
import time as tm

def i_derivative(v, i, r, l):
    return (v - (i * r)) / l

def rk4(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        k1 = f_deriv(v_in[n-1], current_vals[n-1], r, l)
        k2 = f_deriv(v_in[n-1] + dt/2, current_vals[n-1] + dt * k1 / 2, r, l)
        k3 = f_deriv(v_in[n-1] + dt/2, current_vals[n-1] + dt * k2 / 2, r, l)
        k4 = f_deriv(v_in[n], current_vals[n-1] + dt * k3, r, l)
        current_vals[n] = current_vals[n-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return current_vals

def rk2(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        k1 = f_deriv(v_in[n-1], current_vals[n-1], r, l)
        k2 = f_deriv(v_in[n], current_vals[n-1] + dt * k1, r, l)
        current_vals[n] = current_vals[n-1] + dt * (k1 + k2) / 2
    return current_vals

def trapezoidal(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        f1 = f_deriv(v_in[n-1], current_vals[n-1], r, l)
        f2 = f_deriv(v_in[n], current_vals[n-1] + dt * f1, r, l)
        current_vals[n] = current_vals[n-1] + dt * (f1 + f2) / 2
    return current_vals

def backward_euler(init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        current_vals[n] = (current_vals[n-1] * l + dt * v_in[n]) / (l + dt * r)
    return current_vals

def complete_fourier(t, A=1, alpha=0.3, R=1, L=0.5, omega=2*np.pi, N=1000):
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

def square_wave(time, amplitude, period, duty_cycle):
    return np.where((time % period) <= (duty_cycle * period), amplitude, 0)

simulation_time = 10.0
sampling_frequency = 1e3
num_samples = int(simulation_time * sampling_frequency)
time = np.linspace(0, simulation_time, num_samples)
v_in = square_wave(time, amplitude=1, period=1, duty_cycle=0.3)  
r, l, init_current = 1, 0.5, 0

N_baseline = 1000
ground_truth = complete_fourier(time, R=r, L=l, N=N_baseline)

start_rk4 = tm.time()
rk4_result = rk4(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
rk4_time = tm.time() - start_rk4

start_rk2 = tm.time()
rk2_result = rk2(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
rk2_time = tm.time() - start_rk2

start_trap = tm.time()
trapezoidal_result = trapezoidal(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
trapezoidal_time = tm.time() - start_trap

start_be = tm.time()
be_result = backward_euler(init_current, time, v_in, r, l, sampling_frequency)
be_time = tm.time() - start_be

# Compute MAE and MSE
methods = {
    "RK4": rk4_result,
    "RK2": rk2_result,
    "Trapezoidal": trapezoidal_result,
    "Backward Euler": be_result
}

for name, result in methods.items():
    mae = np.mean(np.abs(result - ground_truth))
    mse = np.mean((result - ground_truth) ** 2)
    print(f'{name} MAE: {mae:.12f}, MSE: {mse:.12f}')

print(f'RK4 Time: {rk4_time:.4f}s')
print(f'RK2 Time: {rk2_time:.4f}s')
print(f'Trapezoidal Time: {trapezoidal_time:.4f}s')
print(f'Backward Euler Time: {be_time:.4f}s')
