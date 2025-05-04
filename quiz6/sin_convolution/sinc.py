import numpy as np
import matplotlib.pyplot as plt

T = 1.0
t = np.linspace(-5, 5, 1000)
omega = np.linspace(-100, 100, 1000)

rect = np.where(np.abs(t) <= T/2, 1, 0)

sinc = T * np.sinc(omega * T / (2 * np.pi))

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, rect, label='h(t)')
plt.title('Rectangular Function in Time Domain')
plt.xlabel('Time $t$')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(omega, sinc, color='orange', label='H(s)')
plt.title('Fourier Transform (Sinc Function) in Frequency Domain')
plt.xlabel('Angular Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('ft_box.png')
plt.show()
