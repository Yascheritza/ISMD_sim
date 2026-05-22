import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.stats import linregress

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import cupy as cp
from ismd_sim import make_ismd_state
from tests.test_equal_kernels_nd import (
    tophat_equal_tables_1d,
    half_normal_equal_tables_1d,
    half_normal_equal_tables_2d,
    tophat_equal_tables_2d,
    half_normal_equal_tables_3d,
    tophat_equal_tables_3d,
)
from ismd_sim import make_ismd_state

TIME_STEP = 0.05
DEGREE_BIN = 30.0
OUTPUT_DIR = Path(__file__).resolve().parent / "front_speed_results"


def _regression(times: Sequence[float], values: Sequence[float]) -> Optional[linregress]:
    if len(times) < 2 or len(values) < 2:
        return None
    return linregress(times, values)


def calculate_speed_rates(b_rate, d_rate, sigma_b, sigma_d, sim_type_name, cutoff=5.0, L=40.0, time_limit=40.0):

    if "Type 1" in sim_type_name:
        _, _, r_disp, w_disp = tophat_equal_tables_1d(cutoff=cutoff, sigma=sigma_b)
        _, _, r_comp, w_comp = tophat_equal_tables_1d(cutoff=cutoff, sigma=sigma_d)
    else:
        _, _, r_disp, w_disp = half_normal_equal_tables_1d(cutoff=cutoff, sigma=sigma_b)
        _, _, r_comp, w_comp = half_normal_equal_tables_1d(cutoff=cutoff, sigma=sigma_d)

    kernel_x_disp = np.append(r_disp, cutoff * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    kernel_x_comp = np.append(r_comp, cutoff * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * 1.0

    cells = 2000
    dx = L / cells
    center = 0.5 * L
    xs = (np.arange(cells) + 0.5) * dx
    n_init = np.exp(-((xs - center) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=1, area_size=[L], cell_counts=[cells],
        m=d_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False, n_init=n_init
    )

    threshold = 1e-4
    times = []
    max_radii = []

    dt = 0.05
    steps_per_record = int(1.0 / dt)
    speed = 0.0

    for step_idx in range(int(time_limit)):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            speed = 0.0
            break

        left_idx = active_indices[0]
        right_idx = active_indices[-1]
        left_pos = (left_idx + 0.5) * dx
        right_pos = (right_idx + 0.5) * dx

        max_radius = max(center - left_pos, right_pos - center)

        times.append(sim.time)
        max_radii.append(max_radius)

        if left_pos <= 2.0 or (L - right_pos) <= 2.0:
            break


    if len(active_indices) > 0 and len(times) > 1:
        reg = _regression(times, max_radii)
        speed = reg.slope if reg else 0.0

    sim = None
    cp.get_default_memory_pool().free_all_blocks()

    return speed
# ==============================================================================
# rates diagram generation
# ==============================================================================
def generate_rates_diagram(sim_type_name, fixed_sigma_b, fixed_sigma_d):
    print(f"--- Расчет диаграммы рождаемости/смертности для {sim_type_name} ---")
    print(f"Фиксированные параметры: Sigma_B={fixed_sigma_b}, Sigma_D={fixed_sigma_d}")

    steps = 50
    b_rates = np.linspace(0.5, 2.5, steps)
    d_rates = np.linspace(0.0, 0.4, steps)

    speed_matrix = np.zeros((steps, steps))
    total = steps * steps
    counter = 0

    for i, d_val in enumerate(d_rates):
        for j, b_val in enumerate(b_rates):

            v = calculate_speed_rates(b_rate=b_val, d_rate=d_val,
                                      sigma_b=fixed_sigma_b, sigma_d=fixed_sigma_d,
                                      sim_type_name=sim_type_name)

            speed_matrix[i, j] = v
            cp.get_default_memory_pool().free_all_blocks()

            counter += 1
            sys.stdout.write(f"\r{counter}/{total} | b={b_val:.2f}, d={d_val:.2f} -> v={v:.2f}")

    print("\nГотово. Рисуем...")

    plt.figure(figsize=(9, 7))
    plt.imshow(speed_matrix, origin='lower', cmap='magma', aspect='auto',
               extent=[b_rates.min(), b_rates.max(), d_rates.min(), d_rates.max()])

    plt.colorbar(label="Скорость распространения фронта")
    plt.xlabel("($b$)")
    plt.ylabel("($d$)")
    plt.title(f"Скорость в зависимости от b/d rates")

    filename = f"rates_diagram_{sim_type_name.replace(' ', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #generate_rates_diagram("Type 1", fixed_sigma_b=0.1, fixed_sigma_d=1.0)

    generate_rates_diagram("Type 2", fixed_sigma_b=1.0, fixed_sigma_d=0.1)


if __name__ == "__main__":
    main()

