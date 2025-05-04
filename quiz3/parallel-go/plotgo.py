import matplotlib.pyplot as plt
import numpy as np

# Load RK4 data
rk4_data = np.loadtxt("rk4.txt")
parareal_data = np.loadtxt("parareal.txt")

# Extract time and solution values
t_rk4, i_rk4 = rk4_data[:, 0], rk4_data[:, 1]
t_parareal, i_parareal = parareal_data[:, 0], parareal_data[:, 1]

# Plot both solutions
plt.figure(figsize=(10, 5))
plt.plot(t_rk4, i_rk4, label="RK4 Solution", linestyle="-", color="blue")
plt.plot(t_parareal, i_parareal, label="Parareal RK4 Solution", linestyle="--", color="red")

plt.xlabel("Time (s)")
plt.ylabel("Current (i)")
plt.title("Comparison of RK4 and Parareal RK4 Solutions")
plt.legend()
plt.grid()
plt.savefig("benchmark.png")
plt.show()

