import numpy as np
import matplotlib.pyplot as plt
import time
from numerical_methods import *

# RL circuit differential equation
def i_derivative(v, i, r, l):
    return (v - (i * r)) / l

# Square wave function
def square_wave(time, amplitude, period, duty_cycle):
    return np.where((time % period) <= (duty_cycle * period), amplitude, 0)

# Parameters
r = 10  # Resistance in ohms
l = 1e-3  # Inductance in henries
wave_amplitude = 5 # V
wave_frequency = 500  # Hz
duty_cycle = 0.6 # On time relative to time period of wave
simulation_time = 0.01  # Total simulation time in seconds
sampling_factors = [2**i for i in range(4, 13)]  # Varying sampling factors in powers of 2

# Function to compute global error
def compute_global_error(method, rk4_solution):
    return np.abs(method[-1] - rk4_solution[-1])

errors_dict = {"Forward Euler": [], "Backward Euler": [], "Trapezoidal": [], "RK2": []}
time_dict = {"Forward Euler": [], "Backward Euler": [], "Trapezoidal": [], "RK2": [], "RK4": []}

for sampling_factor in sampling_factors:
    sampling_frequency = wave_frequency * sampling_factor  # Hz
    time_values = np.linspace(0, simulation_time, int(sampling_frequency * simulation_time))
    v_in = square_wave(time_values, amplitude=wave_amplitude, period=1/wave_frequency, duty_cycle=duty_cycle)
    
    # Solve using numerical methods
    init_current = 0
    
    start_time = time.time()
    i_rk4 = rk4(i_derivative, init_current, time_values, v_in, r, l, sampling_frequency)
    time_dict["RK4"].append(time.time() - start_time)
    
    start_time = time.time()
    i_fwd_euler = forward_euler(i_derivative, init_current, time_values, v_in, r, l, sampling_frequency)
    time_dict["Forward Euler"].append(time.time() - start_time)
    
    start_time = time.time()
    i_bwd_euler = backward_euler(init_current, time_values, v_in, r, l, sampling_frequency)
    time_dict["Backward Euler"].append(time.time() - start_time)
    
    start_time = time.time()
    i_trapezoidal = trapezoidal(init_current, time_values, v_in, r, l, sampling_frequency)
    time_dict["Trapezoidal"].append(time.time() - start_time)
    
    start_time = time.time()
    i_rk2 = rk2(i_derivative, init_current, time_values, v_in, r, l, sampling_frequency)
    time_dict["RK2"].append(time.time() - start_time)
    
    # Compute global errors
    errors_dict["Forward Euler"].append(compute_global_error(i_fwd_euler, i_rk4))
    errors_dict["Backward Euler"].append(compute_global_error(i_bwd_euler, i_rk4))
    errors_dict["Trapezoidal"].append(compute_global_error(i_trapezoidal, i_rk4))
    errors_dict["RK2"].append(compute_global_error(i_rk2, i_rk4))

# Function to plot square wave and LTE

def plot_square_wave_and_lte():
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Square wave plot
    axes[0].plot(time_values * 1e3, v_in, label='Square Wave Input')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Square Wave Input Signal')
    axes[0].legend()
    axes[0].grid()
    
    # LTE plot
    axes[1].plot(time_values * 1e3, np.abs(i_rk4 - i_fwd_euler), label='Forward Euler LTE')
    axes[1].plot(time_values * 1e3, np.abs(i_rk4 - i_bwd_euler), label='Backward Euler LTE')
    axes[1].plot(time_values * 1e3, np.abs(i_rk4 - i_trapezoidal), label='Trapezoidal LTE')
    axes[1].plot(time_values * 1e3, np.abs(i_rk4 - i_rk2), label='RK2 LTE')
    
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('LTE')
    axes[1].set_title('Local Truncation Error of Numerical Methods from RK4')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig("../figs/local_error.png")

# Function to plot global error
def plot_global_error():
    plt.figure(figsize=(8, 6))
    for method, errors in errors_dict.items():
        plt.plot(sampling_factors, errors, label=method, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Sampling Factor')
    plt.ylabel('Global Error')
    plt.title('Global Error of Numerical Methods from RK4')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../figs/global_error.png")

# Function to plot computation time
def plot_computation_time():
    plt.figure(figsize=(8, 6))
    for method, times in time_dict.items():
        plt.plot(sampling_factors, times, label=method, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Sampling Factor')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time of Numerical Methods')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../figs/computation_time.png")

# Call plots
plot_square_wave_and_lte()
plot_global_error()
plot_computation_time()
plt.show()
