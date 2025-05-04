import numpy as np

def forward_euler(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    i_vals = np.zeros_like(time)
    i_vals[0] = init_current
    for n in range(1, len(time)):
        i_vals[n] = i_vals[n-1] + f_deriv(v_in[n-1], i_vals[n-1], r, l) / sampling_frequency
    return i_vals

def rk2(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    i_vals = np.zeros_like(time)
    i_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        k1 = f_deriv(v_in[n-1], i_vals[n-1], r, l)
        k2 = f_deriv(v_in[n], i_vals[n-1] + dt * k1, r, l)
        i_vals[n] = i_vals[n-1] + dt * (k1 + k2) / 2
    return i_vals

def trapezoidal(init_current, time, v_in, r, l, sampling_frequency):
    i_vals = np.zeros_like(time)
    i_vals[0] = init_current
    dt = 1 / sampling_frequency

    for n in range(1, len(time)):
        i_vals[n] = (l * i_vals[n-1] + (dt / 2) * (v_in[n-1] - r * i_vals[n-1] + v_in[n])) / (l + (dt * r) / 2)

    return i_vals

def rk4(f_deriv, init_current, time, v_in, r, l, sampling_frequency):
    i_vals = np.zeros_like(time)
    i_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        k1 = f_deriv(v_in[n-1], i_vals[n-1], r, l)
        k2 = f_deriv(v_in[n-1] + dt/2, i_vals[n-1] + dt * k1 / 2, r, l)
        k3 = f_deriv(v_in[n-1] + dt/2, i_vals[n-1] + dt * k2 / 2, r, l)
        k4 = f_deriv(v_in[n], i_vals[n-1] + dt * k3, r, l)
        i_vals[n] = i_vals[n-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return i_vals

def backward_euler(init_current, time, v_in, r, l, sampling_frequency):
    i_vals = np.zeros_like(time)
    i_vals[0] = init_current
    dt = 1 / sampling_frequency
    for n in range(1, len(time)):
        i_vals[n] = (i_vals[n-1] * l + dt * v_in[n]) / (l + dt * r)
    return i_vals