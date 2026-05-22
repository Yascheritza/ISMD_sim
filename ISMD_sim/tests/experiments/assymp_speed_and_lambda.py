import numpy as np
from scipy.optimize import minimize


def c_tophat(lam):
    b = 1.0  # Рождаемость
    m = 0.01  # Смертность
    sigma = 0.1  # Радиус расселения
    return (b * np.sinh(lam * sigma) / (lam * sigma) - m) / lam
def c_gaussian(lam):
    b = 1.0  # Рождаемость
    m = 0.01  # Смертность
    sigma = 1.0  # Радиус расселения
    return (b * np.exp((sigma**2 * lam**2) / 2.0) - m) / lam

res_g = minimize(c_gaussian, x0=1.0, bounds=[(0.01, 10.0)])
theoretical_speed_g = res_g.fun
optimal_lambda_g = res_g.x[0]

res_t = minimize(c_tophat, x0=1.0, bounds=[(0.01, 10.0)])
theoretical_speed_t = res_t.fun
optimal_lambda_t = res_t.x[0]

print(f"Теоретическая скорость, type 1: {theoretical_speed_t:.4f}")
print(f"lambda, type 1: {optimal_lambda_t:.4f}")
print(f"Теоретическая скорость, type 2: {theoretical_speed_g:.4f}")
print(f"Крутизна фронта (lambda), type 2: {optimal_lambda_g:.4f}")




