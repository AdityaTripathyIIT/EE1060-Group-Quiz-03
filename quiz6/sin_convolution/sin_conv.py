import numpy as np
import matplotlib.pyplot as plt

omega = 2 * np.pi / 3
T = 1.0            # Width for symmetric and positive kernel
T1, T2 = 1.0, 2.0  # Shifted kernel range

t_min, t_max = -10, 10
num_points = 2000
t = np.linspace(t_min, t_max, num_points)
dt = t[1] - t[0]

# Signal
sinwt = np.sin(omega * t)

# Kernel generator


def make_box_kernel(t, t_start, t_end):
    kernel = ((t >= t_start) & (t <= t_end)).astype(float)
    kernel /= kernel.sum()
    return kernel


# Kernels
box_symmetric = make_box_kernel(t, -T/2, T/2)
box_positive = make_box_kernel(t, 0, T)
box_shifted = make_box_kernel(t, T1, T2)

# Convolution
conv_sym = np.convolve(sinwt, box_symmetric, mode='same') * dt
conv_pos = np.convolve(sinwt, box_positive, mode='same') * dt
conv_shf = np.convolve(sinwt, box_shifted, mode='same') * dt

# Plotting
plt.figure(figsize=(12, 8))

plt.plot(t, sinwt, label='sin(wt)', color='blue')
plt.plot(t, conv_sym, label='Symmetric Kernel')
plt.title('Convolution with Symmetric Box')
plt.grid(True)
plt.legend()
plt.savefig('sin_symm.png')
plt.show()

plt.plot(t, sinwt, label='sin(wt)', color='blue')
plt.plot(t, conv_pos, label='Positive Kernel')
plt.title('Convolution with Positive Box')
plt.grid(True)
plt.legend()
plt.savefig('sin_pos.png')
plt.show()

plt.plot(t, sinwt, label='sin(wt)', color='blue')
plt.plot(t, conv_shf, label='Shifted Kernel')
plt.title('Convolution with Shifted Box')
plt.grid(True)
plt.legend()
plt.savefig('sin_shift.png')
plt.show()
