import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
import timeit
# Parameters
L = 1.0       # Inductance (H)
R = 1.0       # Resistance (Ohm)
A = 1.0       # Amplitude of square wave
T = 2.0       # Time period of square wave
alpha = 0.5   # Duty ratio (fraction of time V=A)

def V(t):
    return A if (t % T) < (alpha * T) else 0

def dI_dt(t, i):
    return (V(t) - R * i) / L

def coarse_solver(t0, t1, i0, N_coarse):
    t_values = np.linspace(t0, t1, N_coarse + 1)
    dt = (t1 - t0) / N_coarse
    i_values = np.zeros(len(t_values))
    i_values[0] = i0
    for n in range(N_coarse):
        i_values[n+1] = i_values[n] + dt * dI_dt(t_values[n], i_values[n])
    return i_values[-1]

def fine_solver(t0, t1, i0, N_fine):
    t_values = np.linspace(t0, t1, N_fine + 1)
    sol = solve_ivp(dI_dt, [t0, t1], [i0], t_eval=t_values, method="RK45")
    return sol.y[0][-1]

def parareal_parallel(t0, t_final, i0, N_steps, N_coarse, N_fine, max_iter=10):
    t_points = np.linspace(t0, t_final, N_steps + 1)
    dt = (t_final - t0) / N_steps
    i_coarse = np.zeros(N_steps + 1)
    i_fine = np.zeros(N_steps + 1)
    i_corrected = np.zeros(N_steps + 1)

    i_coarse[0] = i_corrected[0] = i0
    for n in range(N_steps):
        i_coarse[n+1] = coarse_solver(t_points[n], t_points[n+1], i_corrected[n], N_coarse)
        i_corrected[n+1] = i_coarse[n+1]

    for k in range(max_iter):
        i_fine[0] = i0
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(fine_solver, t_points[n], t_points[n+1], i_corrected[n], N_fine)
                for n in range(N_steps)
            ]
            fine_results = [f.result() for f in futures]

        for n in range(N_steps):
            i_fine[n+1] = fine_results[n]
        
        max_error = 0
        for n in range(N_steps):
            correction = i_fine[n+1] - i_coarse[n+1]
            i_corrected[n+1] += correction
            max_error = max(max_error, abs(correction))

        print(f"Iteration {k+1}, Max Error: {max_error:.6f}")
        if max_error < 1e-6:
            break
    
    return t_points, i_corrected

def reference_rk4_solution(t0, t_final, i0, N_ref):
    t_values = np.linspace(t0, t_final, N_ref + 1)
    sol = solve_ivp(dI_dt, [t0, t_final], [i0], t_eval=t_values, method="RK45")
    return sol.t, sol.y[0]

t0, t_final = 10, 50
N_steps = 1000     # Parareal time steps
N_coarse = 500     # Steps in coarse solver
N_fine = 5000      # Steps in fine solver
max_iter = 1    # Max iterations for convergence
N_ref = 500000      # High-resolution RK4 for reference

t_parareal, i_parareal = parareal_parallel(t0, t_final, i0=0, N_steps=N_steps, N_coarse=N_coarse, N_fine=N_fine, max_iter=max_iter)
t_rk4, i_rk4 = reference_rk4_solution(t0, t_final, 0, N_ref)
print(timeit.timeit(lambda: parareal_parallel(t0, t_final, i0=0, N_steps=N_steps, N_coarse=N_coarse, N_fine=N_fine, max_iter=max_iter), number=1))
print(timeit.timeit(lambda: reference_rk4_solution(t0, t_final, 0, N_ref), number=1))
plt.figure(figsize=(8,5))
plt.plot(t_parareal, i_parareal, 'b-', label="Parareal Approximation (Parallel)")
plt.plot(t_rk4, i_rk4, 'r-', label="Reference RK4 Solution")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Parareal Algorithm vs Reference RK4 Solution")
plt.legend()
plt.grid()
plt.show()

