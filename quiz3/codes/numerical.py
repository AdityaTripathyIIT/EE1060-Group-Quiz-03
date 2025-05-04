import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def i_derivative(v, i, r, l):
    return (v - (i * r)) / l

def forward_euler(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    for n in range(1, len(time)):
        current_vals[n] = current_vals[n-1] + f_deriv(v_in[n-1], current_vals[n-1], r, l) / sampling_frequency
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

def backward_euler(init_current, time, v_in, r, l, sampling_frequency):
    current_vals = np.zeros_like(time)
    current_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        current_vals[n] = (current_vals[n-1] * l + dt * v_in[n]) / (l + dt * r)
    return current_vals

def square_wave(time, amplitude, period, duty_cycle):
    return np.where((time % period) <= (duty_cycle * period), amplitude, 0)

def update(val):
    r = r_slider.val
    l = l_slider.val
    time_period = wave_period_slider.val
    amplitude = amplitude_slider.val
    duty_cycle = duty_cycle_slider.val
    simulation_time = simulation_time_slider.val
    sampling_frequency = sampling_frequency_slider.val
    init_current = init_current_slider.val

    num_samples = int(simulation_time * sampling_frequency)
    time = np.linspace(0, simulation_time, num_samples)
    v_in = square_wave(time, amplitude, time_period, duty_cycle)

    v_out_fwd_euler = forward_euler(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
    v_out_rk2 = rk2(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
    v_out_trapezoidal = trapezoidal(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
    v_out_rk4 = rk4(i_derivative, init_current, time, v_in, r, l, sampling_frequency)
    v_out_bwd_euler = backward_euler(init_current, time, v_in, r, l, sampling_frequency)

    ax1.clear()
    ax2.clear()

    ax1.plot(time, v_in, label='Square Wave Input')
    ax1.set_title('Square Wave Input Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time, v_out_fwd_euler, label='Forward Euler')
    ax2.plot(time, v_out_rk2, label='RK2')
    ax2.plot(time, v_out_trapezoidal, label='Trapezoidal')
    ax2.plot(time, v_out_rk4, label='RK4')
    ax2.plot(time, v_out_bwd_euler, label='Backward Euler')
    ax2.set_title('Current Responses')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (A)')
    ax2.grid(True)
    ax2.legend()

    fig.canvas.draw_idle()  # Update the main figure properly

# Create figure for plots and sliders in one window
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.3)  # Reduced top space, increased space between plots

# Define slider axes at the bottom of the main figure
ax_r = plt.axes([0.15, 0.15, 0.65, 0.02])
ax_l = plt.axes([0.15, 0.13, 0.65, 0.02])
ax_wave_period = plt.axes([0.15, 0.11, 0.65, 0.02])
ax_amplitude = plt.axes([0.15, 0.09, 0.65, 0.02])
ax_duty_cycle = plt.axes([0.15, 0.07, 0.65, 0.02])
ax_simulation_time = plt.axes([0.15, 0.05, 0.65, 0.02])
ax_sampling_frequency = plt.axes([0.15, 0.03, 0.65, 0.02])
ax_init_current = plt.axes([0.15, 0.01, 0.65, 0.02])

r_slider = Slider(ax_r, 'Resistance', 0.1, 10.0, valinit=1)
l_slider = Slider(ax_l, 'Inductance', 0.1, 2.0, valinit=0.5)
wave_period_slider = Slider(ax_wave_period, 'Wave Period', 0.5, 2.0, valinit=1)
amplitude_slider = Slider(ax_amplitude, 'Amplitude', 0.5, 2.0, valinit=1)
duty_cycle_slider = Slider(ax_duty_cycle, 'Duty Cycle', 0.1, 0.9, valinit=0.5)
simulation_time_slider = Slider(ax_simulation_time, 'Simulation Time', 1, 10, valinit=4)
sampling_frequency_slider = Slider(ax_sampling_frequency, 'Sampling Freq.', 10, 1000, valinit=100)
init_current_slider = Slider(ax_init_current, 'Initial Current', -1.0, 1.0, valinit=0)

for slider in [r_slider, l_slider, wave_period_slider, amplitude_slider, duty_cycle_slider, simulation_time_slider, sampling_frequency_slider, init_current_slider]:
    slider.on_changed(update)

update(None)
plt.show()
