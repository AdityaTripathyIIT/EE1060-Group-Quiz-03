import numpy as np 
import matplotlib.pyplot as plt 
import math
from matplotlib.widgets import Slider


duty_ratio = 0.3
time_period = 1
init_current = 0

resistance = 1
inductance = 2.2e-3

stepsize = 1e-3

time_limits = np.array([0, 100])
t = np.arange(time_limits[0], time_limits[1], stepsize)

def input_voltage(time, T=time_period, alpha=duty_ratio):
    return 1 if (time % T) <= (alpha * T) else 0

def rhsFunc(time, current, L=inductance, R=resistance, alpha=duty_ratio, T=time_period):
    return input_voltage(time, T, alpha) / L - current * R / L

def get_response(L=inductance, R=resistance, h=stepsize, T=time_period, alpha=duty_ratio):
    current_response = []
    current_0 = init_current
    for x in t_parareal:
        current_response.append(current_0)
        k_1 = rhsFunc(x, current_0, L, R, alpha, T)
        k_2 = rhsFunc(x + 0.5 * h, current_0 + 0.5 * h * k_1, L, R, alpha, T)
        k_3 = rhsFunc(x + 0.5 * h, current_0 + 0.5 * h * k_2, L, R, alpha, T)
        k_4 = rhsFunc(x + h, current_0 + h * k_3, L, R, alpha, T)

        current_0 += h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return np.array(current_response)  # Ensure it's a NumPy array for plotting

fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Unpack correctly
plt.subplots_adjust(left=0.25, bottom=0.35)  

response_plot, = ax.plot(t, get_response())

slider_positions = [
    (0.1, 0.25, "DR", 0, 1, duty_ratio),
    (0.1, 0.20, "T", 0.1, 10, time_period),
    (0.1, 0.15, "R", 0.1, 100, resistance),
    (0.5, 0.25, "L", 1e-4, 10, inductance),
    (0.5, 0.20, "h", 1e-6, 1e-3, stepsize),
]

sliders = {}
for x_pos, y_pos, label, min_val, max_val, init_val in slider_positions:
    ax_slider = plt.axes([x_pos, y_pos, 0.35, 0.03])
    sliders[label] = Slider(ax_slider, label, min_val, max_val, valinit=init_val)

def update(val):
    alpha = sliders["DR"].val
    T = sliders["T"].val
    R = sliders["R"].val
    L = sliders["L"].val
    h = sliders["h"].val
    
    global t  
    t = np.arange(time_limits[0], time_limits[1], h)  
    response_plot.set_xdata(t)
    response_plot.set_ydata(get_response(L, R, h, T, alpha))
    ax.relim()  
    ax.autoscale_view()  
    fig.canvas.draw_idle()

for slider in sliders.values():
    slider.on_changed(update)

plt.show()

