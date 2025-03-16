import numpy as np
import matplotlib.pyplot as plt

# Given differential equation: L di/dt + Ri = V(t)

# Values of L and R
L = 1.0
R = 1.0 

# Sampled Time 
T = 2  # Total duration (seconds)
fs = 1000  # High sampling frequency
dt = 1 / fs  # Time step

# Time vector
t = np.arange(0, T, dt)

# Given V(t)
v_t = np.sin(2 * np.pi * 50 * t) #+ np.cos(2 * np.pi * 50 * t)

# Applying Nyquist Sampling
f_max = 50  # Maximum frequency in the signal
f_ny = 3 * f_max  # Oversampling (chosen arbitrarily)
t_ny = np.arange(0, T, 1 / f_ny)  # Corrected Nyquist-sampled time vector
v_ny = np.sin(2 * np.pi * 50 * t_ny) #+ np.cos(2 * np.pi * 50 * t_ny)

# Compute the FFT of V(t)
V_FFT = np.fft.fft(v_ny)
frequencies = np.fft.fftfreq(len(t_ny), 1/f_ny)  # Frequency bins

# Finding I_f for the given differential equation
denominator = R + 1j * 2 * np.pi * frequencies * L
denominator[frequencies == 0] = R  # Avoid division by zero at DC
I_f = V_FFT / denominator

# Converting Frequency domain to Time domain
i_t = np.fft.ifft(I_f).real  # Take only the real part

# Plot the functions
plt.figure(figsize=(10, 4))
plt.plot(t, v_t, label="Original V(t)", alpha=0.6)
plt.scatter(t_ny, v_ny, color='red', label="Nyquist Sampled Points", marker='x')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 0.25)
plt.grid()
plt.legend()
plt.title("Voltage and Current Response")
plt.show()

# Second figure: Current plot
plt.figure(figsize=(10, 4))
plt.plot(t_ny, i_t, label="Recovered I(t)", linestyle='dashed', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.ylim(-0.005 , 0.005)
plt.xlim(0,0.25)
plt.grid()
plt.legend()
plt.title("Current Response")
plt.show()
