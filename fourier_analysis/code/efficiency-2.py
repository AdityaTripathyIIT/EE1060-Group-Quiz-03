import math

def compute_energy(A, alpha, T, R, L):
    I_cy = (A / R) * math.exp(-R * T / L) * ((math.exp(R * alpha * T / L) - 1) / (1 - math.exp(-R * T / L)))
    
    I_alpha_T = (A / R) + (I_cy - (A / R)) * math.exp(-R * alpha * T / L)

    exponent = -2 * R * (T - alpha * T) / L
    E_2 = (L / 2) * (I_alpha_T ** 2) * (1 - math.exp(exponent)) 
    E_1 = ((A ** 2) / R) * (alpha * T) \
        + 2 * A * (I_cy - (A / R)) * (L / R) * (1 - math.exp(-R * alpha * T / L)) \
        + ((I_cy - (A / R)) ** 2) * (L / 2) * (1 - math.exp(-2 * R * alpha * T / L))
    
    E_total = E_1 + E_2
    
    term1 = (A / R) * alpha * T
    term2 = (I_cy - A / R) * (L / R) * (1 - math.exp(-R * alpha * T / L))
    term3 = I_alpha_T * (L / R) * (1 - math.exp(-R * (T - alpha * T) / L))
    
    E_s = A * (term1 + term2)
    return E_total/E_s

A = 10
alpha = 0.7
T = 1e-6
R = 1e3
L = 1e3

E_total = compute_energy(A, alpha, T, R, L)
E_source = A * alpha * T
efficiency = E_total/E_source
print(f"Amplitude = {A}")
print(f"alpha = {alpha}")
print(f"Time Period = {T}")
print(f"Resistance = {R}")
print(f"Inductance = {L}")
print(f"Efficiency = {E_total}")

