import numpy as np

rk4_data = np.loadtxt("rk4.txt")
parareal_data = np.loadtxt("parareal.txt")

rk4_dict = {t: v for t, v in rk4_data}
parareal_dict = {t: v for t, v in parareal_data}

common_times = sorted(set(rk4_dict.keys()) & set(parareal_dict.keys()))

errors = [abs(rk4_dict[t] - parareal_dict[t]) for t in common_times]

mean_error = np.mean(errors)

print(f"Mean Error: {mean_error}")
