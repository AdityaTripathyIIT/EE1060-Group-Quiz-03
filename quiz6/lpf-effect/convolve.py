import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
t = np.linspace(-5, 5, 1000, endpoint=False)


def func(x):
    if np.abs(x) > 1:
        return 0
    elif (x < -0.5):
        return 1 - np.abs(x-0.5)
    elif (x < 0.25):
        return np.exp(-x)
    else:
        return np.sin(2*x)


signal = np.array([func(x) for x in t])

T = 1
rect = np.where(np.abs(t) <= T / 2, 1, 0)

convolved = np.convolve(signal, rect, mode='same') / rect.sum()

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, rect, label='Rectangular Pulse')
plt.title('Original Signal and Rectangular Pulse')
plt.legend()
plt.grid(True)
plt.savefig('func.png')
plt.show()
plt.figure(figsize=(10, 4))
plt.plot(t, convolved, label='Filtered Signal (Low-pass effect)')
plt.title('Convolved Signal (Low-pass Effect)')
plt.legend()
plt.grid(True)
plt.savefig('conv.png')
plt.show()


N = len(t)
freqs = fftfreq(N, d=(t[1] - t[0]))
orig_spec = np.abs(fft(signal))
conv_spec = np.abs(fft(convolved))

plt.figure(figsize=(10, 4))
plt.plot(freqs, orig_spec, label='Original Spectrum')
plt.plot(freqs, conv_spec, label='Filtered Spectrum')
plt.title('Frequency Response')
plt.legend()
plt.grid(True)
plt.savefig('frequency_response.png')
plt.show()
