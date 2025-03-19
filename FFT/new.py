import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from matplotlib.widgets import Slider

def nyquist_sampling_rate(T, harmonics=5):
    f_fundamental = 1 / T
    f_max = (2 * harmonics - 1) * f_fundamental
    return   2*f_max

def generate_square_wave(T, t, duty=0.5, amplitude=1.0):
    f = 1 / T
    return amplitude * (np.mod(t * f, 1) < duty).astype(float)

def solve_rl_circuit(R, L, v_t, t):
    N = len(t)
    dt = t[1] - t[0]  # Time step
    V_w = fft(v_t)  # FFT of v(t)
    frequencies = fftfreq(N, d=dt) * 2 * np.pi  # Angular frequencies (ω)
    # Transfer function in the frequency domain
    H_w = 1 / (R + 1j * frequencies * L)
    H_w[0] = 1 / R  # DC component handling to avoid division by zero
    # Solve for I(w) in the frequency domain
    I_w = V_w * H_w
    # Inverse FFT to get i(t)
    i_t = np.real(ifft(I_w))
    return i_t

def update_plot(val):
    R = R_slider.val
    L = L_slider.val
    T = T_slider.val
    harmonics = harmonics_slider.val
    duty = duty_slider.val
    amplitude = amplitude_slider.val
    t_max = t_max_slider.val
    
    nyquist_rate = nyquist_sampling_rate(T, harmonics)
    N_fine = 5000
    t_fine = np.linspace(0, t_max, N_fine, endpoint=False)
    v_t_fine = 10*generate_square_wave(T, t_fine, duty, amplitude)
    N_nyquist = int(t_max * nyquist_rate)
    t_nyquist = np.linspace(0, t_max, N_nyquist, endpoint=False)
    v_t_nyquist = 10*generate_square_wave(T, t_nyquist, duty, amplitude)
    i_t_fine = solve_rl_circuit(R, L, v_t_fine, t_fine)
    i_t_nyquist = solve_rl_circuit(R, L, v_t_nyquist, t_nyquist)
    
    axs[0].cla()
    axs[0].plot(t_fine, v_t_fine, color="black")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_title("High-Resolution Square Wave")
    axs[0].grid()
    
    axs[1].cla()
    axs[1].plot(t_nyquist, v_t_nyquist, 'o-', color="r", markersize=3)
    axs[1].set_ylabel("Voltage (V)")
    axs[1].set_title("Nyquist-Sampled Square Wave")
    axs[1].grid()
    
    axs[2].cla()
    axs[2].plot(t_fine, i_t_fine, linewidth=2, color="b")
    axs[2].set_ylabel("Current (A)")
    axs[2].set_title("Current Response i(t) (High Res)")
    axs[2].grid()
    
    axs[3].cla()
    axs[3].plot(t_nyquist, i_t_nyquist, 'o-', linewidth=2, color="g", markersize=3)
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Current (A)") 
    axs[3].set_title("Current Response i(t) (Nyquist-Sampled)")
    axs[3].grid()
    
    fig.canvas.draw_idle()

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.50)

ax_R = plt.axes([0.1, 0.35, 0.8, 0.03])
ax_L = plt.axes([0.1, 0.30, 0.8, 0.03])
ax_T = plt.axes([0.1, 0.25, 0.8, 0.03])
ax_harmonics = plt.axes([0.1, 0.20, 0.8, 0.03])
ax_duty = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_amplitude = plt.axes([0.1, 0.10, 0.8, 0.03])
ax_t_max = plt.axes([0.1, 0.05, 0.8, 0.03])

R_slider = Slider(ax_R, "Resistance (Ω)", 0.1, 10, valinit=1)
L_slider = Slider(ax_L, "Inductance (H)", 0.1, 10, valinit=1)
T_slider = Slider(ax_T, "Period (s)", 0.5, 5, valinit=2)
harmonics_slider = Slider(ax_harmonics, "Harmonics", 1, 10, valinit=5, valstep=1)
duty_slider = Slider(ax_duty, "Duty Cycle", 0.1, 0.9, valinit=0.5)
amplitude_slider = Slider(ax_amplitude, "Amplitude", 0.1, 5, valinit=1.0)
t_max_slider = Slider(ax_t_max, "Simulation Time (s)", 1, 10, valinit=5)

R_slider.on_changed(update_plot)
L_slider.on_changed(update_plot)
T_slider.on_changed(update_plot)
harmonics_slider.on_changed(update_plot)
duty_slider.on_changed(update_plot)
amplitude_slider.on_changed(update_plot)
t_max_slider.on_changed(update_plot)

update_plot(None)
plt.show()

