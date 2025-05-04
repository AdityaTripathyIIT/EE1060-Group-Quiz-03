import numpy as np
import matplotlib.pyplot as plt

num_points = 500
x = np.linspace(0, 4 * np.pi, num_points)
clean_signal = np.sin(x)

# Add Gaussian noise to create a noisy signal
noise_level = 0.3
noise = np.random.normal(0, noise_level, num_points)
noisy_signal = clean_signal + noise

kernel_size = 10
rectangular_kernel = np.ones(kernel_size) / kernel_size

denoised_signals = [noisy_signal]
current_signal = noisy_signal

# Perform 10 passes and store results
for i in range(10):
    current_signal = np.convolve(
        current_signal, rectangular_kernel, mode='same')
    denoised_signals.append(current_signal)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(x, noisy_signal, label='Noisy Signal')
plt.plot(x, clean_signal, label='Clean Signal', linestyle='--', alpha=0.7)
plt.title('Noisy Signal')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, denoised_signals[1], label='1st Pass', color='orange')
plt.plot(x, clean_signal, label='Clean Signal', linestyle='--', alpha=0.7)
plt.title('After 1 Pass of Rectangular Kernel')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, denoised_signals[5], label='5th Pass', color='green')
plt.plot(x, clean_signal, label='Clean Signal', linestyle='--', alpha=0.7)
plt.title('After 5 Passes of Rectangular Kernel')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, denoised_signals[10], label='10th Pass', color='red')
plt.plot(x, clean_signal, label='Clean Signal', linestyle='--', alpha=0.7)
plt.title('After 10 Passes of Rectangular Kernel')
plt.legend()

plt.tight_layout()
plt.show()


def calculate_mse(signal, reference):
    return np.mean((signal - reference) ** 2)


mse_values = [calculate_mse(signal, clean_signal)
              for signal in denoised_signals]

print(f"MSE of noisy signal: {mse_values[0]:.6f}")
print(f"MSE after 1 pass: {mse_values[1]:.6f}")
print(f"MSE after 5 passes: {mse_values[5]:.6f}")
print(f"MSE after 10 passes: {mse_values[10]:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(range(11), mse_values, marker='o')
plt.xlabel('Number of Passes')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Number of Rectangular Kernel Passes')
plt.grid(True)
plt.show()
