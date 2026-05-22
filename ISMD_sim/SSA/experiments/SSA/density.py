import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SSA import make_ssa_state_1d
from tests.test_equal_kernels_nd import tophat_equal_tables_1d, half_normal_equal_tables_1d

OUTPUT_DIR = Path(__file__).resolve().parent / "advisor_report_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_strict_experiment(sim_type="Type 2", max_retries=10):
    L = 200.0
    CENTER = L / 2.0
    START_N = 50

    if sim_type == "Type 1":
        sigma_b = 0.1
        sigma_d = 1.0
        dd_rate = 0.5
    else:  # Type 2
        sigma_b = 1.0
        sigma_d = 0.1
        dd_rate = 1.0

    cutoff = 5.0
    res_b = half_normal_equal_tables_1d(cutoff, sigma=sigma_b)
    res_d = half_normal_equal_tables_1d(cutoff, sigma=sigma_d)

    for attempt in range(max_retries):
        print(f"[{sim_type}] Попытка {attempt + 1}...")

        sim = make_ssa_state_1d(
            M=1, area_len=L, cell_count=4000,
            birth_rates=np.array([1.0]),
            death_rates=np.array([0.01]),
            dd_matrix=np.array([[dd_rate]]),
            birth_x=res_b[0], birth_y=res_b[1],
            death_x=res_d[2], death_y=res_d[3],
            cutoffs=np.array([[cutoff]]),
            seed=np.random.randint(1000000),
            is_periodic=False
        )

        initial_coords = np.random.normal(loc=CENTER, scale=1.0, size=START_N)
        for pos in initial_coords:

            if 0 < pos < L:
                sim.spawn_particle(0, pos)

        times = []
        populations = []
        max_coords = []
        snapshots = {}

        dt = 0.5
        curr_t = 0.0
        last_snapshot_time = 0.0
        snapshot_interval = 20.0

        success = False

        while True:
            sim.run_until_time(curr_t + dt)
            curr_t = sim.current_time()
            count = int(sim.current_population())

            if count == 0:
                print("  -> Вымирание (Pop=0).")
                break

            xs = np.array([sim.positions[i, 0] for i in range(count)])

            times.append(curr_t)
            populations.append(count)
            max_coords.append(np.max(xs))

            if curr_t - last_snapshot_time >= snapshot_interval:
                snapshots[round(curr_t)] = xs.copy()
                last_snapshot_time = curr_t

            # Проверка границ (10 и 190)
            if np.min(xs) <= 10.0 or np.max(xs) >= 190.0:
                print(f"  -> УСПЕХ! Граница достигнута на t={curr_t:.1f}")
                snapshots[round(curr_t)] = xs.copy()
                success = True
                break

            if curr_t > 4000:  # Тайм-аут
                print("  -> Слишком долго.")
                break

        if success:
            return times, populations, max_coords, snapshots, sim_type, L

    return None


def plot_results(times, populations, max_coords, snapshots, sim_type, L):
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations, color='green', lw=2)
    plt.title(f"[{sim_type}], half normal, численность")
    plt.xlabel("Время")
    plt.ylabel("Численность")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_pop.png")

    cut = int(len(times) * 0.3)
    res = linregress(times[cut:], max_coords[cut:])

    plt.figure(figsize=(8, 4))
    plt.plot(times, max_coords, label="Фронт")
    plt.plot(times, [res.intercept + res.slope * t for t in times], 'r--',
             label=f"linear fit: v={res.slope:.4f}")
    plt.title(f"[{sim_type}], half normal, Скорость распространения фронта")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_speed.png")

    plt.figure(figsize=(10, 6))

    sorted_times = sorted(snapshots.keys())
    if len(sorted_times) > 6:
        indices = np.linspace(0, len(sorted_times) - 1, 6, dtype=int)
        times_to_plot = [sorted_times[i] for i in indices]
    else:
        times_to_plot = sorted_times

    for t in times_to_plot:
        plt.hist(snapshots[t], bins=50, density=True, histtype='step',
                 linewidth=2, label=f"t={t}s", range=(0, L))  # range фиксирует оси!

    plt.xlim(0, L)
    plt.title(f"[{sim_type}] , half normal, плотность с течением времени")
    plt.xlabel("Координата")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_density.png")
    plt.show()


if __name__ == "__main__":
    data = run_strict_experiment("Type 2")
    if data: plot_results(*data)

    #data_t1 = run_strict_experiment("Type 1")
    #if data_t1: plot_results(*data_t1)