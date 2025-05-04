import numpy as np
import matplotlib.pyplot as plt

T = 2  # Width of the pulse
t = np.linspace(-3*T, 3*T, 500)

y = np.piecewise(t, [t < -T, (-T <= t) & (t < T), t >= T],
                 [0, lambda t: t + T, 2*T])

plt.figure(figsize=(6, 4))
plt.plot(t, y, label=r'$y(t)$', color='b')
plt.axvline(x=-T, linestyle='--', color='gray')
plt.axvline(x=T, linestyle='--', color='gray')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.title("Convolution of Symmetric Pulse")
plt.legend()
plt.grid(True)
plt.savefig('step_symm.png')
plt.show()

T = 2  # Width of the pulse
t = np.linspace(-T, 3*T, 500)

y = np.piecewise(t, [t < 0, (0 <= t) & (t < T), t >= T], [0, lambda t: t, T])

plt.figure(figsize=(6, 4))
plt.plot(t, y, label=r'$y(t)$', color='g')
plt.axvline(x=0, linestyle='--', color='gray')
plt.axvline(x=T, linestyle='--', color='gray')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.title("Convolution of Pulse from $0$ to $T$")
plt.legend()
plt.grid(True)
plt.savefig('step_pos.png')
plt.show()

t_vals = np.linspace(-3, 3, 500)


def y_function(t, T, t0):
    if t < -T + t0:
        return 0
    elif -T + t0 <= t < T + t0:
        return t + T - t0
    else:
        return 2*T


T_numeric = 2
t0_numeric = 1

y_vals = [y_function(t, T_numeric, t0_numeric) for t in t_vals]

plt.figure(figsize=(6, 4))
plt.plot(t_vals, y_vals, label=r'$y(t)$', color='r')

plt.axvline(x=-T_numeric + t0_numeric, linestyle='--', color='gray')
plt.axvline(x=T_numeric + t0_numeric, linestyle='--', color='gray')

plt.hlines(y=2*T_numeric, xmin=T_numeric + t0_numeric,
           xmax=max(t_vals), color='r', linestyle='-')

plt.text(-T_numeric + t0_numeric - 0.2, 0, r'$0$', fontsize=12, color='black')
plt.text((T_numeric + t0_numeric)/2, T_numeric,
         r'$t + T - t_0$', fontsize=12, color='black')
plt.text(T_numeric + t0_numeric + 0.2, 2*T_numeric,
         r'$2T$', fontsize=12, color='black')

plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.title("Convolution of Shifted Pulse")
plt.legend()
plt.grid(True)
plt.savefig('step_shifted.png')
plt.show()
