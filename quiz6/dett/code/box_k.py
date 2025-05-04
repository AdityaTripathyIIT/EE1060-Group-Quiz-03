import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
import argparse


def create_convolution_plot(num_repetitions=4, T=0.5, τ=1.0, t_range=(-4, 4),
                            num_points=1000, show_kernel=True, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 12

    t = np.linspace(t_range[0], t_range[1], num_points)
    dt = t[1] - t[0]

    # Triangle pulse centered at 0, width = 2*τ, height = 1/τ for unit area
    triangle_pulse = np.zeros_like(t)
    triangle_pulse = np.maximum(1 - np.abs(t / τ), 0) / τ

    # Box kernel: uniform over [-T, T] with height 1/(2T)
    box_kernel = np.zeros_like(t)
    box_kernel[(t >= -T) & (t <= T)] = 1 / (2 * T)

    current_signal = triangle_pulse.copy()

    for _ in range(num_repetitions):
        current_signal = signal.convolve(
            current_signal, box_kernel * dt, mode='same')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Final convolution result
    ax.plot(t, current_signal, color='blue', linewidth=2,
            label=f'{num_repetitions}× Convolution Result')

    # Original triangle pulse
    ax.plot(t, triangle_pulse, color='orange', linewidth=2,
            label='Original Triangle Pulse')

    # Show kernel if enabled
    if show_kernel:
        kernel_line = np.zeros_like(t)
        kernel_line[(t >= -T) & (t <= T)] = 1 / (2 * T)
        ax.plot(t, kernel_line, color='green', linewidth=2,
                label=f'Box Kernel $h(t)$ (T={T})')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Convolution of Triangle Pulse with Box Kernel ({num_repetitions}×)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(-0.1, max(1.1, np.max(current_signal) + 0.1))

    if num_repetitions >= 3:
        ax.annotate('Approaches\nSmooth Curve',
                    xy=(0, 0.5),
                    xytext=(-2.5, 0.7),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)

    ax.text(t_range[0] + 0.5, -0.07,
            r'$y_n(t) = y_{n-1}(t) * h(t)$, where $h(t) = \frac{1}{2T}$ for $|t| \leq T$',
            fontsize=10)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', framealpha=1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate convolution plot with triangle pulse')
    parser.add_argument('--repetitions', type=int, default=4,
                        help='Number of convolutions to perform (default: 4)')
    parser.add_argument('--kernel-width', type=float, default=0.5,
                        help='Half-width of box kernel (T value) (default: 0.5)')
    parser.add_argument('--pulse-width', type=float, default=1.0,
                        help='Half-width (τ) of the triangle pulse (default: 1.0)')
    parser.add_argument('--show-kernel', action='store_true',
                        help='Show the kernel in the plot')
    parser.add_argument('--output', type=str, default='triangle_convolution_example.png',
                        help='Output file path (default: triangle_convolution_example.png)')

    args = parser.parse_args()

    fig, ax = create_convolution_plot(
        num_repetitions=args.repetitions,
        T=args.kernel_width,
        τ=args.pulse_width,
        show_kernel=args.show_kernel,
        save_path=args.output
    )

    plt.show()
