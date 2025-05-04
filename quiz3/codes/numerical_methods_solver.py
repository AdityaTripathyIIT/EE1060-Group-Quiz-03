import numpy as np
from numerical_methods import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')

def i_derivative(v, i, r, l):
    return (v - (i * r)) / l

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
    time_int = np.linspace(0, simulation_time, num_samples)
    
    display_sf = 1000 * (1 / time_period)
    num_display_samples = int(simulation_time * display_sf)
    time_disp = np.linspace(0, simulation_time, num_display_samples)
    
    v_in_disp = square_wave(time_disp, amplitude, time_period, duty_cycle)
    v_in_int = square_wave(time_int, amplitude, time_period, duty_cycle)
    
    i_fwd_euler = forward_euler(i_derivative, init_current, time_int, v_in_int, r, l, sampling_frequency)
    i_rk2 = rk2(i_derivative, init_current, time_int, v_in_int, r, l, sampling_frequency)
    i_trapezoidal = trapezoidal(init_current, time_int, v_in_int, r, l, sampling_frequency)
    i_rk4 = rk4(i_derivative, init_current, time_int, v_in_int, r, l, sampling_frequency)
    i_bwd_euler = backward_euler(init_current, time_int, v_in_int, r, l, sampling_frequency)
    
    ax1.clear()
    ax2.clear()
    
    ax1.plot(time_disp, v_in_disp, label='Square Wave Input')
    ax1.set_title('Square Wave Input Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(time_int, i_fwd_euler, label='Forward Euler')
    ax2.plot(time_int, i_rk2, label='RK2')
    ax2.plot(time_int, i_trapezoidal, label='Trapezoidal')
    ax2.plot(time_int, i_rk4, label='RK4')
    ax2.plot(time_int, i_bwd_euler, label='Backward Euler')
    ax2.set_title('Current Responses')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (A)')
    ax2.grid(True)
    ax2.legend()
    
    fig.canvas.draw_idle()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.3)

ax_r = plt.axes([0.15, 0.15, 0.65, 0.02])
ax_l = plt.axes([0.15, 0.13, 0.65, 0.02])
ax_wave_period = plt.axes([0.15, 0.11, 0.65, 0.02])
ax_amplitude = plt.axes([0.15, 0.09, 0.65, 0.02])
ax_duty_cycle = plt.axes([0.15, 0.07, 0.65, 0.02])
ax_simulation_time = plt.axes([0.15, 0.05, 0.65, 0.02])
ax_sampling_frequency = plt.axes([0.15, 0.03, 0.65, 0.02])
ax_init_current = plt.axes([0.15, 0.01, 0.65, 0.02])

r_slider = Slider(ax_r, 'Resistance', 0.1, 10.0, valinit=1)
l_slider = Slider(ax_l, 'Inductance', 0.1, 10.0, valinit=1)
wave_period_slider = Slider(ax_wave_period, 'Wave Period', 0.5, 2.0, valinit=1)
amplitude_slider = Slider(ax_amplitude, 'Amplitude', 0.5, 2.0, valinit=1)
duty_cycle_slider = Slider(ax_duty_cycle, 'Duty Cycle', 0.1, 0.9, valinit=0.5)
simulation_time_slider = Slider(ax_simulation_time, 'Simulation Time', 1, 10, valinit=4)
sampling_frequency_slider = Slider(ax_sampling_frequency, 'Sampling Freq.', 2, 100, valinit=10)
init_current_slider = Slider(ax_init_current, 'Initial Current', -1.0, 1.0, valinit=0)

for slider in [r_slider, l_slider, wave_period_slider, amplitude_slider, duty_cycle_slider,
               simulation_time_slider, sampling_frequency_slider, init_current_slider]:
    slider.on_changed(update)

update(None)
plt.show()
