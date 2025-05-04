import numpy as np
import matplotlib.pyplot as plt

a = -1.0
T = 1.0
T1, T2 = 1.0, 2.0

t_min, t_max = -10, 10
num_points = 2000
t = np.linspace(t_min, t_max, num_points)
dt = t[1] - t[0]

exp_at = np.exp(a * t) * (t >= 0)


def make_box_kernel(t, t_start, t_end):
    kernel = ((t >= t_start) & (t <= t_end)).astype(float)
    kernel /= np.abs(t_start - t_end)
    return kernel


box_symmetric = make_box_kernel(t, -T/2, T/2)
box_positive = make_box_kernel(t, 0, T)
box_shifted = make_box_kernel(t, T1, T2)

conv_sym = np.convolve(exp_at, box_symmetric, mode='same') * dt
conv_pos = np.convolve(exp_at, box_positive, mode='same') * dt
conv_shf = np.convolve(exp_at, box_shifted, mode='same') * dt

plt.figure(figsize=(12, 6))
plt.plot(t, exp_at, label='e^{a·t}')
plt.plot(t, conv_sym, label='Symmetric Kernel')
plt.title('Convolution with Symmetric Box')
plt.grid(True)
plt.legend()
plt.savefig('exp_symm.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, exp_at, label='e^{a·t}')
plt.plot(t, conv_pos, label='Positive Kernel')
plt.title('Convolution with Positive Box')
plt.grid(True)
plt.legend()
plt.savefig('exp_pos.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, exp_at, label='e^{a·t}')
plt.plot(t, conv_shf, label='Shifted Kernel')
plt.title('Convolution with Shifted Box')
plt.grid(True)
plt.legend()
plt.savefig('exp_shift.png')
plt.show()
