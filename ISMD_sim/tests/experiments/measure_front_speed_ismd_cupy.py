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

from tests.test_equal_kernels_ismd_cupy import (
    half_normal_equal_tables_1d,
    half_normal_equal_tables_2d,
    half_normal_equal_tables_3d, tophat_equal_tables_1d, tophat_equal_tables_2d, tophat_equal_tables_3d,
)


TIME_STEP = 1.0
DEGREE_BIN = 30.0
OUTPUT_DIR = Path(__file__).resolve().parent / "front_speed_results"


def _regression(times: Sequence[float], values: Sequence[float]) -> Optional[linregress]:
    if len(times) < 2 or len(values) < 2:
        return None
    return linregress(times, values)


def _plot_max_distance(times: Sequence[float], values: Sequence[float], reg, title: str, ylabel: str, output: Path) -> None:
    if len(times) == 0:
        return
    time_arr = np.asarray(times, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(time_arr, value_arr, marker="o", linewidth=1.5, label="Observed")

    if reg is not None:
        fitted = reg.intercept + reg.slope * time_arr
        ax.plot(time_arr, fitted, linestyle="--", linewidth=1.5, label=f"Linear fit (slope={reg.slope:.4f})")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_left_right(
    times: Sequence[float],
    left: Sequence[float],
    right: Sequence[float],
    reg_left,
    reg_right,
    output: Path,
) -> None:
    if len(times) == 0:
        return
    time_arr = np.asarray(times, dtype=np.float64)
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(time_arr, left_arr, marker="o", label="Left front")
    ax.plot(time_arr, right_arr, marker="s", label="Right front")

    if reg_left is not None:
        ax.plot(
            time_arr,
            reg_left.intercept + reg_left.slope * time_arr,
            linestyle="--",
            color=ax.lines[0].get_color(),
            label=f"Left fit (slope={reg_left.slope:.4f})",
        )
    if reg_right is not None:
        ax.plot(
            time_arr,
            reg_right.intercept + reg_right.slope * time_arr,
            linestyle="--",
            color=ax.lines[1].get_color(),
            label=f"Right fit (slope={reg_right.slope:.4f})",
        )

    ax.set_title("1D front positions (left/right)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance from center")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_2d_polygons(polygons: list[np.ndarray], times: Sequence[float], extent: tuple[float, float, float, float], output: Path) -> None:
    if not polygons or len(times) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    time_arr = np.asarray(times, dtype=np.float64)
    order_desc = np.argsort(time_arr)[::-1]  # draw later times first so earlier times overlay them
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=time_arr.min(), vmax=time_arr.max())
    for idx in order_desc:
        poly = polygons[idx]
        color = cmap(norm(time_arr[idx]))
        ax.fill(
            poly[:, 0],
            poly[:, 1],
            color=color,
            alpha=0.4,
            edgecolor="k",
            linewidth=0.7,
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D population border over time")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="Time")
    fig.savefig(output, dpi=220)
    plt.close(fig)


from ismd_sim import make_ismd_state  # Не забудьте обновить импорт!


def measure_front_speed_1d_type1_halfnormal(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    # Среднее поле: достаточно для графика, быстро для матриц
    L = 40.0
    cells = 3200
    boundary_margin = 2.0

    _, _, r_disp, w_disp = half_normal_equal_tables_1d(cutoff=5.0, sigma=0.1)
    kernel_x_disp = np.append(r_disp, 5.0 * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    _, _, r_comp, w_comp = half_normal_equal_tables_1d(cutoff=5.0, sigma=1.0)
    kernel_x_comp = np.append(r_comp, 5.0 * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    dx = L / cells
    center = 0.5 * L
    xs = (np.arange(cells) + 0.5) * dx
    n_init = np.exp(-((xs - center) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=1, area_size=[L], cell_counts=[cells],
        m=m_rate, birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False, n_init=n_init
    )

    threshold = 1e-4
    times, max_radii, left_radii, right_radii = [], [], [], []

    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))

    for step_idx in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            print("Внимание: Популяция вымерла или ушла в NaN!")
            break

        left_idx = active_indices[0]
        right_idx = active_indices[-1]

        left_pos = (left_idx + 0.5) * dx
        right_pos = (right_idx + 0.5) * dx

        max_radius = max(center - left_pos, right_pos - center)
        left_dist = center - left_pos
        right_dist = right_pos - center

        times.append(sim.time)
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        # Индикатор пульса в консоли, чтобы вы не переживали
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

        if left_pos <= boundary_margin or (L - right_pos) <= boundary_margin:
            print("Волна достигла границы! Завершаю...")
            break

    time_arr = np.asarray(times, dtype=np.float64)
    return {
        "times": time_arr,
        "max": np.asarray(max_radii, dtype=np.float64),
        "left": np.asarray(left_radii, dtype=np.float64),
        "right": np.asarray(right_radii, dtype=np.float64),
        "reg_max": _regression(time_arr, np.asarray(max_radii, dtype=np.float64)),
        "reg_left": _regression(time_arr, np.asarray(left_radii, dtype=np.float64)),
        "reg_right": _regression(time_arr, np.asarray(right_radii, dtype=np.float64)),
    }


def measure_front_speed_1d_type2_halfnormal(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    # Среднее поле: достаточно для графика, быстро для матриц
    L = 40.0
    cells = 3200
    boundary_margin = 2.0

    _, _, r_disp, w_disp = half_normal_equal_tables_1d(cutoff=5.0, sigma=1.0)
    kernel_x_disp = np.append(r_disp, 5.0 * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    _, _, r_comp, w_comp = half_normal_equal_tables_1d(cutoff=5.0, sigma=0.1)
    kernel_x_comp = np.append(r_comp, 5.0 * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    # УЗКАЯ ГАУССИАНА (чтобы не касалась краев на старте)
    dx = L / cells
    center = 0.5 * L
    xs = (np.arange(cells) + 0.5) * dx
    n_init = np.exp(-((xs - center) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=1, area_size=[L], cell_counts=[cells],
        m=m_rate, birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False, n_init=n_init
    )

    threshold = 1e-4
    times, max_radii, left_radii, right_radii = [], [], [], []

    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))

    for step_idx in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            print("Внимание: Популяция вымерла или ушла в NaN!")
            break

        left_idx = active_indices[0]
        right_idx = active_indices[-1]

        left_pos = (left_idx + 0.5) * dx
        right_pos = (right_idx + 0.5) * dx

        max_radius = max(center - left_pos, right_pos - center)
        left_dist = center - left_pos
        right_dist = right_pos - center

        times.append(sim.time)
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        # Индикатор пульса в консоли, чтобы вы не переживали
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

        if left_pos <= boundary_margin or (L - right_pos) <= boundary_margin:
            print("Волна достигла границы! Завершаю...")
            break

    time_arr = np.asarray(times, dtype=np.float64)
    return {
        "times": time_arr,
        "max": np.asarray(max_radii, dtype=np.float64),
        "left": np.asarray(left_radii, dtype=np.float64),
        "right": np.asarray(right_radii, dtype=np.float64),
        "reg_max": _regression(time_arr, np.asarray(max_radii, dtype=np.float64)),
        "reg_left": _regression(time_arr, np.asarray(left_radii, dtype=np.float64)),
        "reg_right": _regression(time_arr, np.asarray(right_radii, dtype=np.float64)),
    }


def measure_front_speed_1d_type1_tophat(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    # Среднее поле: достаточно для графика, быстро для матриц
    L = 40.0
    cells = 3200
    boundary_margin = 2.0

    _, _, r_disp, w_disp = tophat_equal_tables_1d(cutoff=5.0, sigma=0.1)
    kernel_x_disp = np.append(r_disp, 5.0 * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    _, _, r_comp, w_comp = tophat_equal_tables_1d(cutoff=5.0, sigma=1.0)
    kernel_x_comp = np.append(r_comp, 5.0 * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    dx = L / cells
    center = 0.5 * L
    xs = (np.arange(cells) + 0.5) * dx
    n_init = np.exp(-((xs - center) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=1, area_size=[L], cell_counts=[cells],
        m=m_rate, birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False, n_init=n_init
    )

    threshold = 1e-4
    times, max_radii, left_radii, right_radii = [], [], [], []

    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))

    for step_idx in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            print("Внимание: Популяция вымерла или ушла в NaN!")
            break

        left_idx = active_indices[0]
        right_idx = active_indices[-1]

        left_pos = (left_idx + 0.5) * dx
        right_pos = (right_idx + 0.5) * dx

        max_radius = max(center - left_pos, right_pos - center)
        left_dist = center - left_pos
        right_dist = right_pos - center

        times.append(sim.time)
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        # Индикатор пульса в консоли, чтобы вы не переживали
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

        if left_pos <= boundary_margin or (L - right_pos) <= boundary_margin:
            print("Волна достигла границы! Завершаю...")
            break

    time_arr = np.asarray(times, dtype=np.float64)
    return {
        "times": time_arr,
        "max": np.asarray(max_radii, dtype=np.float64),
        "left": np.asarray(left_radii, dtype=np.float64),
        "right": np.asarray(right_radii, dtype=np.float64),
        "reg_max": _regression(time_arr, np.asarray(max_radii, dtype=np.float64)),
        "reg_left": _regression(time_arr, np.asarray(left_radii, dtype=np.float64)),
        "reg_right": _regression(time_arr, np.asarray(right_radii, dtype=np.float64)),
    }


def measure_front_speed_1d_type2_tophat(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    L = 40.0
    cells = 3200
    boundary_margin = 2.0

    _, _, r_disp, w_disp = tophat_equal_tables_1d(cutoff=5.0, sigma=1.0)
    kernel_x_disp = np.append(r_disp, 5.0 * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    _, _, r_comp, w_comp = tophat_equal_tables_1d(cutoff=5.0, sigma=0.1)
    kernel_x_comp = np.append(r_comp, 5.0 * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    # УЗКАЯ ГАУССИАНА (чтобы не касалась краев на старте)
    dx = L / cells
    center = 0.5 * L
    xs = (np.arange(cells) + 0.5) * dx
    n_init = np.exp(-((xs - center) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=1, area_size=[L], cell_counts=[cells],
        m=m_rate, birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False, n_init=n_init
    )

    threshold = 1e-4
    times, max_radii, left_radii, right_radii = [], [], [], []

    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))

    for step_idx in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            print("Внимание: Популяция вымерла или ушла в NaN!")
            break

        left_idx = active_indices[0]
        right_idx = active_indices[-1]

        left_pos = (left_idx + 0.5) * dx
        right_pos = (right_idx + 0.5) * dx

        max_radius = max(center - left_pos, right_pos - center)
        left_dist = center - left_pos
        right_dist = right_pos - center

        times.append(sim.time)
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        # Индикатор пульса в консоли, чтобы вы не переживали
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

        if left_pos <= boundary_margin or (L - right_pos) <= boundary_margin:
            print("Волна достигла границы! Завершаю...")
            break

    time_arr = np.asarray(times, dtype=np.float64)
    return {
        "times": time_arr,
        "max": np.asarray(max_radii, dtype=np.float64),
        "left": np.asarray(left_radii, dtype=np.float64),
        "right": np.asarray(right_radii, dtype=np.float64),
        "reg_max": _regression(time_arr, np.asarray(max_radii, dtype=np.float64)),
        "reg_left": _regression(time_arr, np.asarray(left_radii, dtype=np.float64)),
        "reg_right": _regression(time_arr, np.asarray(right_radii, dtype=np.float64)),
    }

def measure_front_speed_2d_type1_halfnormal(time_step: float = TIME_STEP):
    # --- СТРОГИЕ ПАРАМЕТРЫ ИЗ СТАТЬИ ---
    b_rate = 1.0  # c+ = 1
    m_rate = 0.01  # m = 0.01
    dd_rate = 1.0  # c- = 1

    # Размер поля L=10, сетка 80x80 (N_total = 6400). Шаг dx = 0.125
    Lx, Ly = 10, 10
    Nx, Ny = 80, 80

    # Type 1: Top-hat ядра. Рождение короткое (0.1), конкуренция дальняя (1.0)
    cutoff_disp = 1.0
    _, _, r_disp, w_disp = half_normal_equal_tables_2d(cutoff=cutoff_disp, sigma=0.1)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.5
    _, _, r_comp, w_comp = half_normal_equal_tables_2d(cutoff=cutoff_comp, sigma=1.0)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy = 0.5 * Lx, 0.5 * Ly
    dx, dy = Lx / float(Nx), Ly / float(Ny)
    populations: list[float] = []
    snapshots: dict[float, np.ndarray] = {}
    h_vol = dx * dy  # Площадь одной ячейки
    mid_y_idx = Ny // 2  # Индекс центральной строки по оси Y
    # --- НАЧАЛЬНОЕ УСЛОВИЕ (2D Гауссиана c_0=1, sigma_0=1) ---
    n_init = np.zeros(Nx * Ny, dtype=np.float64)
    for iy in range(Ny):
        for ix in range(Nx):
            x = (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            idx = iy * Nx + ix
            n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=2,
        area_size=[Lx, Ly],
        cell_counts=[Nx, Ny],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    angular_bins_history: list[np.ndarray] = []
    polygons: list[np.ndarray] = []

    num_bins = int(360 / DEGREE_BIN)
    angles = np.deg2rad(np.arange(0, 360, DEGREE_BIN))
    angles_ext = np.append(angles, angles[0])

    threshold = 1e-4
    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        iy = active_indices // Nx
        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy

        diff_x = xs - cx
        diff_y = ys - cy
        radii = np.hypot(diff_x, diff_y)
        max_radius = float(radii.max())

        phi = (np.degrees(np.arctan2(diff_y, diff_x)) + 360.0) % 360.0
        indices = np.floor(phi / DEGREE_BIN).astype(int) % num_bins
        bin_max = np.zeros(num_bins, dtype=np.float64)
        for idx, r in zip(indices, radii):
            if r > bin_max[idx]:
                bin_max[idx] = r

        radii_ext = np.append(bin_max, bin_max[0])
        polygon = np.column_stack((cx + radii_ext * np.cos(angles_ext), cy + radii_ext * np.sin(angles_ext)))

        times.append(sim.time)
        max_radii.append(max_radius)
        total_pop = np.sum(n) * h_vol
        populations.append(float(total_pop))

        # 2. Делаем срез: берем строку матрицы, проходящую ровно через центр (mid_y_idx)
        n_2d = n.reshape((Ny, Nx))  # Превращаем плоский массив обратно в 2D матрицу
        snapshots[sim.time] = n_2d[mid_y_idx, :].copy()
        angular_bins_history.append(bin_max)
        polygons.append(polygon)

        # Выход, если коснулись границы
        # Выход, если коснулись границы
        boundary_reached = (
                xs.min() <= cutoff_max
                or (Lx - xs.max()) <= cutoff_max
                or ys.min() <= cutoff_max
                or (Ly - ys.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            angular_bins_history.pop()
            polygons.pop()

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            populations.pop()  # Удаляем лишнюю численность
            del snapshots[sim.time]  # Удаляем лишний срез плотности
            # -------------------------

            break
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "angular_bins": angular_bins_history,
        "polygons": polygons,
        "reg_max": _regression(time_arr, max_arr),
        "cutoff": cutoff_max,
        "extent": (0.0, Lx, 0.0, Ly),
        # НОВЫЕ КЛЮЧИ:
        "populations": populations,
        "snapshots": snapshots
    }


def measure_front_speed_2d_type2_halfnormal(time_step: float = TIME_STEP):
    # --- СТРОГИЕ ПАРАМЕТРЫ ИЗ СТАТЬИ ---
    b_rate = 1.0  # c+ = 1
    m_rate = 0.01  # m = 0.01
    dd_rate = 1.0  # c- = 1

    Lx, Ly = 20, 20
    Nx, Ny = 80, 80

    # Type 2: Гауссианы (Half-normal). Рождение дальнее (1.0), конкуренция короткая (0.1)
    cutoff_disp = 1.5
    _, _, r_disp, w_disp = half_normal_equal_tables_2d(cutoff=cutoff_disp, sigma=1.0)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.0
    _, _, r_comp, w_comp = half_normal_equal_tables_2d(cutoff=cutoff_comp, sigma=0.1)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy = 0.5 * Lx, 0.5 * Ly
    dx, dy = Lx / float(Nx), Ly / float(Ny)
    populations: list[float] = []
    snapshots: dict[float, np.ndarray] = {}
    h_vol = dx * dy  # Площадь одной ячейки
    mid_y_idx = Ny // 2  # Индекс центральной строки по оси Y
    # --- НАЧАЛЬНОЕ УСЛОВИЕ (2D Гауссиана c_0=1, sigma_0=1) ---
    n_init = np.zeros(Nx * Ny, dtype=np.float64)
    # В начале функции measure_front...
    for iy in range(Ny):
        for ix in range(Nx):
            x = (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            idx = iy * Nx + ix
            n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=2,
        area_size=[Lx, Ly],
        cell_counts=[Nx, Ny],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    angular_bins_history: list[np.ndarray] = []
    polygons: list[np.ndarray] = []

    num_bins = int(360 / DEGREE_BIN)
    angles = np.deg2rad(np.arange(0, 360, DEGREE_BIN))
    angles_ext = np.append(angles, angles[0])

    threshold = 1e-4
    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        iy = active_indices // Nx
        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy

        diff_x = xs - cx
        diff_y = ys - cy
        radii = np.hypot(diff_x, diff_y)
        max_radius = float(radii.max())

        phi = (np.degrees(np.arctan2(diff_y, diff_x)) + 360.0) % 360.0
        indices = np.floor(phi / DEGREE_BIN).astype(int) % num_bins
        bin_max = np.zeros(num_bins, dtype=np.float64)
        for idx, r in zip(indices, radii):
            if r > bin_max[idx]:
                bin_max[idx] = r

        radii_ext = np.append(bin_max, bin_max[0])
        polygon = np.column_stack((cx + radii_ext * np.cos(angles_ext), cy + radii_ext * np.sin(angles_ext)))

        times.append(sim.time)
        max_radii.append(max_radius)
        # Существующий код:
        # times.append(sim.time)
        # max_radii.append(max_radius)

        # НОВЫЙ КОД ДЛЯ СБОРА ДАННЫХ:
        # 1. Считаем общий интеграл плотности: Сумма всех ячеек * площадь ячейки
        total_pop = np.sum(n) * h_vol
        populations.append(float(total_pop))

        # 2. Делаем срез: берем строку матрицы, проходящую ровно через центр (mid_y_idx)
        n_2d = n.reshape((Ny, Nx))  # Превращаем плоский массив обратно в 2D матрицу
        snapshots[sim.time] = n_2d[mid_y_idx, :].copy()
        angular_bins_history.append(bin_max)
        polygons.append(polygon)

        # Выход, если коснулись границы
        # Выход, если коснулись границы
        boundary_reached = (
                xs.min() <= cutoff_max
                or (Lx - xs.max()) <= cutoff_max
                or ys.min() <= cutoff_max
                or (Ly - ys.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            angular_bins_history.pop()
            polygons.pop()

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            populations.pop()  # Удаляем лишнюю численность
            del snapshots[sim.time]  # Удаляем лишний срез плотности
            # -------------------------

            break
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")
    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "angular_bins": angular_bins_history,
        "polygons": polygons,
        "reg_max": _regression(time_arr, max_arr),
        "cutoff": cutoff_max,
        "extent": (0.0, Lx, 0.0, Ly),
        # НОВЫЕ КЛЮЧИ:
        "populations": populations,
        "snapshots": snapshots
    }


def measure_front_speed_2d_type1_tophat(time_step: float = TIME_STEP):
    # --- СТРОГИЕ ПАРАМЕТРЫ ИЗ СТАТЬИ ---
    b_rate = 1.0  # c+ = 1
    m_rate = 0.01  # m = 0.01
    dd_rate = 1.0  # c- = 1

    # Размер поля L=10, сетка 80x80 (N_total = 6400). Шаг dx = 0.125
    Lx, Ly = 10, 10
    Nx, Ny = 80, 80

    # Type 1: Top-hat ядра. Рождение короткое (0.1), конкуренция дальняя (1.0)
    cutoff_disp = 1.0
    _, _, r_disp, w_disp = tophat_equal_tables_2d(cutoff=cutoff_disp, sigma=0.1)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.5
    _, _, r_comp, w_comp = tophat_equal_tables_2d(cutoff=cutoff_comp, sigma=1.0)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy = 0.5 * Lx, 0.5 * Ly
    dx, dy = Lx / float(Nx), Ly / float(Ny)
    populations: list[float] = []
    snapshots: dict[float, np.ndarray] = {}
    h_vol = dx * dy  # Площадь одной ячейки
    mid_y_idx = Ny // 2  # Индекс центральной строки по оси Y
    # --- НАЧАЛЬНОЕ УСЛОВИЕ (2D Гауссиана c_0=1, sigma_0=1) ---
    n_init = np.zeros(Nx * Ny, dtype=np.float64)
    for iy in range(Ny):
        for ix in range(Nx):
            x = (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            idx = iy * Nx + ix
            n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=2,
        area_size=[Lx, Ly],
        cell_counts=[Nx, Ny],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    angular_bins_history: list[np.ndarray] = []
    polygons: list[np.ndarray] = []

    num_bins = int(360 / DEGREE_BIN)
    angles = np.deg2rad(np.arange(0, 360, DEGREE_BIN))
    angles_ext = np.append(angles, angles[0])

    threshold = 1e-4
    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        iy = active_indices // Nx
        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy

        diff_x = xs - cx
        diff_y = ys - cy
        radii = np.hypot(diff_x, diff_y)
        max_radius = float(radii.max())

        phi = (np.degrees(np.arctan2(diff_y, diff_x)) + 360.0) % 360.0
        indices = np.floor(phi / DEGREE_BIN).astype(int) % num_bins
        bin_max = np.zeros(num_bins, dtype=np.float64)
        for idx, r in zip(indices, radii):
            if r > bin_max[idx]:
                bin_max[idx] = r

        radii_ext = np.append(bin_max, bin_max[0])
        polygon = np.column_stack((cx + radii_ext * np.cos(angles_ext), cy + radii_ext * np.sin(angles_ext)))

        times.append(sim.time)
        max_radii.append(max_radius)
        total_pop = np.sum(n) * h_vol
        populations.append(float(total_pop))

        # 2. Делаем срез: берем строку матрицы, проходящую ровно через центр (mid_y_idx)
        n_2d = n.reshape((Ny, Nx))  # Превращаем плоский массив обратно в 2D матрицу
        snapshots[sim.time] = n_2d[mid_y_idx, :].copy()
        angular_bins_history.append(bin_max)
        polygons.append(polygon)

        # Выход, если коснулись границы
        # Выход, если коснулись границы
        boundary_reached = (
                xs.min() <= cutoff_max
                or (Lx - xs.max()) <= cutoff_max
                or ys.min() <= cutoff_max
                or (Ly - ys.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            angular_bins_history.pop()
            polygons.pop()

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            populations.pop()  # Удаляем лишнюю численность
            del snapshots[sim.time]  # Удаляем лишний срез плотности
            # -------------------------

            break
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "angular_bins": angular_bins_history,
        "polygons": polygons,
        "reg_max": _regression(time_arr, max_arr),
        "cutoff": cutoff_max,
        "extent": (0.0, Lx, 0.0, Ly),
        # НОВЫЕ КЛЮЧИ:
        "populations": populations,
        "snapshots": snapshots
    }


def measure_front_speed_2d_type2_tophat(time_step: float = TIME_STEP):
    # --- СТРОГИЕ ПАРАМЕТРЫ ИЗ СТАТЬИ ---
    b_rate = 1.0  # c+ = 1
    m_rate = 0.01  # m = 0.01
    dd_rate = 1.0  # c- = 1

    Lx, Ly = 20, 20
    Nx, Ny = 80, 80

    # Type 2: Гауссианы (Half-normal). Рождение дальнее (1.0), конкуренция короткая (0.1)
    cutoff_disp = 1.5
    _, _, r_disp, w_disp = tophat_equal_tables_2d(cutoff=cutoff_disp, sigma=1.0)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.0
    _, _, r_comp, w_comp = tophat_equal_tables_2d(cutoff=cutoff_comp, sigma=0.1)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy = 0.5 * Lx, 0.5 * Ly
    dx, dy = Lx / float(Nx), Ly / float(Ny)
    populations: list[float] = []
    snapshots: dict[float, np.ndarray] = {}
    h_vol = dx * dy  # Площадь одной ячейки
    mid_y_idx = Ny // 2  # Индекс центральной строки по оси Y
    # --- НАЧАЛЬНОЕ УСЛОВИЕ (2D Гауссиана c_0=1, sigma_0=1) ---
    n_init = np.zeros(Nx * Ny, dtype=np.float64)
    # В начале функции measure_front...
    for iy in range(Ny):
        for ix in range(Nx):
            x = (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            idx = iy * Nx + ix
            n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.2 ** 2))

    sim = make_ismd_state(
        ndim=2,
        area_size=[Lx, Ly],
        cell_counts=[Nx, Ny],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    angular_bins_history: list[np.ndarray] = []
    polygons: list[np.ndarray] = []

    num_bins = int(360 / DEGREE_BIN)
    angles = np.deg2rad(np.arange(0, 360, DEGREE_BIN))
    angles_ext = np.append(angles, angles[0])

    threshold = 1e-4
    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        iy = active_indices // Nx
        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy

        diff_x = xs - cx
        diff_y = ys - cy
        radii = np.hypot(diff_x, diff_y)
        max_radius = float(radii.max())

        phi = (np.degrees(np.arctan2(diff_y, diff_x)) + 360.0) % 360.0
        indices = np.floor(phi / DEGREE_BIN).astype(int) % num_bins
        bin_max = np.zeros(num_bins, dtype=np.float64)
        for idx, r in zip(indices, radii):
            if r > bin_max[idx]:
                bin_max[idx] = r

        radii_ext = np.append(bin_max, bin_max[0])
        polygon = np.column_stack((cx + radii_ext * np.cos(angles_ext), cy + radii_ext * np.sin(angles_ext)))

        times.append(sim.time)
        max_radii.append(max_radius)
        # Существующий код:
        # times.append(sim.time)
        # max_radii.append(max_radius)

        # НОВЫЙ КОД ДЛЯ СБОРА ДАННЫХ:
        # 1. Считаем общий интеграл плотности: Сумма всех ячеек * площадь ячейки
        total_pop = np.sum(n) * h_vol
        populations.append(float(total_pop))

        # 2. Делаем срез: берем строку матрицы, проходящую ровно через центр (mid_y_idx)
        n_2d = n.reshape((Ny, Nx))  # Превращаем плоский массив обратно в 2D матрицу
        snapshots[sim.time] = n_2d[mid_y_idx, :].copy()
        angular_bins_history.append(bin_max)
        polygons.append(polygon)

        # Выход, если коснулись границы
        # Выход, если коснулись границы
        boundary_reached = (
                xs.min() <= cutoff_max
                or (Lx - xs.max()) <= cutoff_max
                or ys.min() <= cutoff_max
                or (Ly - ys.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            angular_bins_history.pop()
            polygons.pop()

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            populations.pop()  # Удаляем лишнюю численность
            del snapshots[sim.time]  # Удаляем лишний срез плотности
            # -------------------------

            break
        print(f"Time: {sim.time:.1f} | Max Radius: {max_radius:.2f}")
    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "angular_bins": angular_bins_history,
        "polygons": polygons,
        "reg_max": _regression(time_arr, max_arr),
        "cutoff": cutoff_max,
        "extent": (0.0, Lx, 0.0, Ly),
        # НОВЫЕ КЛЮЧИ:
        "populations": populations,
        "snapshots": snapshots
    }


def measure_front_speed_3d_type1(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    # БЕЗОПАСНАЯ 3D СЕТКА: 16x16x16 (Матрицы ~134 MB)
    Lx, Ly, Lz = 10.0, 10.0, 10.0
    Nx, Ny, Nz = 10, 10, 10

    # Увеличенный Type 1 для 3D: Узкое рождение (2.0), Дальняя конкуренция (6.0)
    cutoff_disp = 1.0
    _, _, r_disp, w_disp = tophat_equal_tables_3d(cutoff=cutoff_disp, sigma=0.1)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.5
    _, _, r_comp, w_comp = tophat_equal_tables_3d(cutoff=cutoff_comp, sigma=1.0)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    dx, dy, dz = Lx / float(Nx), Ly / float(Ny), Lz / float(Nz)

    # 3D Гауссиана в центре поля (с увеличенной начальной шириной sigma_0 = 2.0)
    n_init = np.zeros(Nx * Ny * Nz, dtype=np.float64)
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                x = (ix + 0.5) * dx
                y = (iy + 0.5) * dy
                z = (iz + 0.5) * dz
                idx = (iz * Ny + iy) * Nx + ix
                n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2.0 * 0.8 ** 2))

    sim = make_ismd_state(
        ndim=3,
        area_size=[Lx, Ly, Lz],
        cell_counts=[Nx, Ny, Nz],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    threshold = 1e-4

    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        temp = active_indices // Nx
        iy = temp % Ny
        iz = temp // Ny

        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy
        zs = (iz + 0.5) * dz

        diff_x = xs - cx
        diff_y = ys - cy
        diff_z = zs - cz
        radii = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
        max_radius = float(radii.max())

        times.append(sim.time)
        max_radii.append(max_radius)

        boundary_reached = (
                xs.min() <= cutoff_max or (Lx - xs.max()) <= cutoff_max or
                ys.min() <= cutoff_max or (Ly - ys.max()) <= cutoff_max or
                zs.min() <= cutoff_max or (Lz - zs.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "reg_max": _regression(time_arr, max_arr),
    }


def measure_front_speed_3d_type2(time_step: float = TIME_STEP):
    b_rate = 1.0
    m_rate = 0.01
    dd_rate = 1.0

    Lx, Ly, Lz = 10.0, 10.0, 10.0
    Nx, Ny, Nz = 10, 10, 10

    # Увеличенный Type 2: Гауссианы. Дальнее рождение (6.0), узкая конкуренция (2.0)
    cutoff_disp = 1.5
    _, _, r_disp, w_disp = half_normal_equal_tables_3d(cutoff=cutoff_disp, sigma=1.0)
    kernel_x_disp = np.append(r_disp, cutoff_disp * 1.01)
    kernel_w_disp = np.append(w_disp, 0.0)

    cutoff_comp = 1.0
    _, _, r_comp, w_comp = half_normal_equal_tables_3d(cutoff=cutoff_comp, sigma=0.1)
    kernel_x_comp = np.append(r_comp, cutoff_comp * 1.01)
    kernel_w_comp = np.append(w_comp, 0.0)

    birth_y_scaled = kernel_w_disp * b_rate
    death_y_scaled = kernel_w_comp * dd_rate

    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    dx, dy, dz = Lx / float(Nx), Ly / float(Ny), Lz / float(Nz)

    n_init = np.zeros(Nx * Ny * Nz, dtype=np.float64)
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                x = (ix + 0.5) * dx
                y = (iy + 0.5) * dy
                z = (iz + 0.5) * dz
                idx = (iz * Ny + iy) * Nx + ix
                n_init[idx] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2.0 * 0.8 ** 2))

    sim = make_ismd_state(
        ndim=3,
        area_size=[Lx, Ly, Lz],
        cell_counts=[Nx, Ny, Nz],
        m=m_rate,
        birth_x=kernel_x_disp, birth_y=birth_y_scaled,
        death_x=kernel_x_comp, death_y=death_y_scaled,
        periodic=False,
        n_init=n_init
    )

    times: list[float] = []
    max_radii: list[float] = []
    threshold = 1e-4
    dt = 0.05
    steps_per_record = max(1, int(time_step / dt))
    cutoff_max = max(cutoff_disp, cutoff_comp)

    for _ in range(500):
        for _ in range(steps_per_record):
            sim.step(dt)

        n = sim.get_density()
        active_indices = np.where(n > threshold)[0]
        if len(active_indices) == 0:
            break

        ix = active_indices % Nx
        temp = active_indices // Nx
        iy = temp % Ny
        iz = temp // Ny

        xs = (ix + 0.5) * dx
        ys = (iy + 0.5) * dy
        zs = (iz + 0.5) * dz

        diff_x = xs - cx
        diff_y = ys - cy
        diff_z = zs - cz
        radii = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
        max_radius = float(radii.max())

        times.append(sim.time)
        max_radii.append(max_radius)

        boundary_reached = (
                xs.min() <= cutoff_max or (Lx - xs.max()) <= cutoff_max or
                ys.min() <= cutoff_max or (Ly - ys.max()) <= cutoff_max or
                zs.min() <= cutoff_max or (Lz - zs.max()) <= cutoff_max
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "reg_max": _regression(time_arr, max_arr),
    }


def _plot_log_difference_left_right(times: Sequence[float], left: Sequence[float], right: Sequence[float], title: str,
                                    output: Path) -> None:
    if len(times) == 0:
        return
    time_arr = np.asarray(times, dtype=np.float64)
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)

    # Вычисляем абсолютную разницу между фронтами
    diff = np.abs(left_arr - right_arr)
    # Защита от строгих нулей, так как log(0) выдаст ошибку.
    # Заменяем нули на крошечное число (машинный эпсилон).
    diff_safe = np.where(diff == 0, 1e-16, diff)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(time_arr, diff_safe, marker="x", linestyle="-", color="red", label="|Left - Right|")

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute Difference (Log Scale)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)

def plot_ismd_results_2d(times, populations, max_coords, snapshots, sim_type, Lx, Nx):
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    import numpy as np

    # 1. Численность (Интеграл плотности по всему полю)
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations, color='green', lw=2)
    plt.title(f"[{sim_type}] Динамика общей численности (Интеграл)")
    plt.xlabel("Время")
    plt.ylabel("Численность")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_pop.png")

    # 2. Скорость (Регрессия)
    cut = int(len(times) * 0.3)
    res = linregress(times[cut:], max_coords[cut:])

    plt.figure(figsize=(8, 4))
    plt.plot(times, max_coords, label="Фронт", marker='o', markersize=4)
    if res is not None:
        plt.plot(times, [res.intercept + res.slope * t for t in times], 'r--',
                    label=f"linear fit: v={res.slope:.4f}")
    plt.title(f"[{sim_type}] Максимальный радиус фронта")
    plt.xlabel("Время")
    plt.ylabel("Радиус")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_speed.png")

    # 3. --- ИСПРАВЛЕНИЕ: Срез плотности через центр поля ---
    plt.figure(figsize=(10, 6))

    sorted_times = sorted(snapshots.keys())
    if len(sorted_times) > 6:
        indices = np.linspace(0, len(sorted_times) - 1, 6, dtype=int)
        times_to_plot = [sorted_times[i] for i in indices]
    else:
        times_to_plot = sorted_times

    # Координаты X для отрисовки среза
    dx = Lx / float(Nx)
    xs = (np.arange(Nx) + 0.5) * dx

    for t in times_to_plot:
        density_slice = snapshots[t]
        # Обычный plot, так как у нас уже есть точные значения плотности в каждой ячейке
        plt.plot(xs, density_slice, linewidth=2, label=f"t={t:.1f}s")

    plt.xlim(0, Lx)
    plt.title(f"[{sim_type}] Профиль плотности (срез через центр)")
    plt.xlabel("Координата X")
    plt.ylabel("Плотность n(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / f"{sim_type}_density.png")
    plt.close()  # Обязательно закрываем, чтобы не засорять память

def _plot_log_deviations_from_fit(times: Sequence[float], values: Sequence[float], reg, title: str,
                                  output: Path) -> None:
    if len(times) == 0 or reg is None:
        return
    time_arr = np.asarray(times, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)

    # Вычисляем идеальную прямую
    fitted = reg.intercept + reg.slope * time_arr

    # Вычисляем невязку (разницу между реальной позицией и идеальной прямой)
    diff = np.abs(value_arr - fitted)
    diff_safe = np.where(diff <= 1e-16, 1e-16, diff)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(time_arr, diff_safe, marker="d", linestyle="-", color="purple", label="|Observed - Linear Fit|")

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Deviation from Fit (Log Scale)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Запускаю 1D ISMD симуляцию...")
    res_1d_t1_h = measure_front_speed_1d_type1_halfnormal()
    res_1d_t2_h = measure_front_speed_1d_type2_halfnormal()
    res_1d_t1_t = measure_front_speed_1d_type1_tophat()
    res_1d_t2_t = measure_front_speed_1d_type2_tophat()

    print("Запускаю 2D ISMD симуляцию...")
    res_2d_t1_h = measure_front_speed_2d_type1_halfnormal()
    res_2d_t2_h = measure_front_speed_2d_type2_halfnormal()
    res_2d_t1_t = measure_front_speed_2d_type1_tophat()
    res_2d_t2_t = measure_front_speed_2d_type2_tophat()

    # print("Запускаю 3D ISMD симуляцию...")
    # res_3d_t1 = measure_front_speed_3d_type1()
    # res_3d_t2 = measure_front_speed_3d_type2()

    # 1D plots
    _plot_max_distance(
        res_1d_t1_h["times"], res_1d_t1_h["max"], res_1d_t1_h["reg_max"],
        "1D ISMD type 1 half normal kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "1d_ismd_max_distance_t1_h.png",
    )
    _plot_max_distance(
        res_1d_t1_t["times"], res_1d_t1_t["max"], res_1d_t1_t["reg_max"],
        "1D ISMD type 1 tophat kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "1d_ismd_max_distance_t1_t.png",
    )
    _plot_max_distance(
        res_1d_t2_h["times"], res_1d_t2_h["max"], res_1d_t2_h["reg_max"],
        "1D ISMD type 2 half normal kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "1d_ismd_max_distance_t2_h.png",
     )
    _plot_max_distance(
        res_1d_t2_t["times"], res_1d_t2_t["max"], res_1d_t2_t["reg_max"],
        "1D ISMD type 2 tophat kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "1d_ismd_max_distance_t2_t.png",
    )
    _plot_left_right(
        res_1d_t1_h["times"], res_1d_t1_h["left"], res_1d_t1_h["right"],
        res_1d_t1_h["reg_left"], res_1d_t1_h["reg_right"],
        OUTPUT_DIR / "1d_ismd_half_normal_kernels_left_right_distance_t1.png",
    )
    _plot_left_right(
        res_1d_t1_t["times"], res_1d_t1_t["left"], res_1d_t1_t["right"],
        res_1d_t1_t["reg_left"], res_1d_t1_t["reg_right"],
        OUTPUT_DIR / "1d_ismd_tophat_kernels_left_right_distance_t1.png",
    )
    _plot_left_right(
        res_1d_t2_h["times"], res_1d_t2_h["left"], res_1d_t2_h["right"],
        res_1d_t2_h["reg_left"], res_1d_t2_h["reg_right"],
        OUTPUT_DIR / "1d_ismd_half_normal_kernels_left_right_distance_t2.png",
    )
    _plot_left_right(
        res_1d_t2_t["times"], res_1d_t2_t["left"], res_1d_t2_t["right"],
        res_1d_t2_t["reg_left"], res_1d_t2_t["reg_right"],
        OUTPUT_DIR / "1d_ismd_tophat_kernels_left_right_distance_t2.png",
    )

    #2D plots
    _plot_max_distance(
        res_2d_t1_h["times"], res_2d_t1_h["max"], res_2d_t1_h["reg_max"],
        "2D ISMD type 1 half normal kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "2d_ismd_h_max_distance_t1.png",
    )
    _plot_max_distance(
        res_2d_t1_t["times"], res_2d_t1_t["max"], res_2d_t1_t["reg_max"],
        "2D ISMD type 1 tophat kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "2d_ismd_t_max_distance_t1.png",
    )
    _plot_max_distance(
        res_2d_t2_h["times"], res_2d_t2_h["max"], res_2d_t2_h["reg_max"],
        "2D ISMD type 2 half normal kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "2d_ismd_h_max_distance_t2.png",
    )
    _plot_max_distance(
        res_2d_t2_t["times"], res_2d_t2_t["max"], res_2d_t2_t["reg_max"],
        "2D ISMD type 2 tophat kernels front max distance vs time", "Max distance from center",
        OUTPUT_DIR / "2d_ismd_max_distance_t2.png",
    )
    _plot_2d_polygons(
        res_2d_t1_h["polygons"], res_2d_t1_h["times"], res_2d_t1_h["extent"],
        OUTPUT_DIR / "2d_ismd_h_border_polygons_t1.png",
    )
    _plot_2d_polygons(
        res_2d_t1_t["polygons"], res_2d_t1_t["times"], res_2d_t1_t["extent"],
        OUTPUT_DIR / "2d_ismd_t_border_polygons_t1.png",
    )
    _plot_2d_polygons(
        res_2d_t2_h["polygons"], res_2d_t2_h["times"], res_2d_t2_h["extent"],
        OUTPUT_DIR / "2d_ismd_h_border_polygons_t2.png",
    )
    _plot_2d_polygons(
        res_2d_t2_t["polygons"], res_2d_t2_t["times"], res_2d_t2_t["extent"],
        OUTPUT_DIR / "2d_ismd_t_border_polygons_t2.png",
    )
    plot_ismd_results_2d(
        times=res_2d_t2_h["times"],
        populations=res_2d_t2_h["populations"],
        max_coords=res_2d_t2_h["max"],
        snapshots=res_2d_t2_h["snapshots"],
        sim_type="type2_h_20x20",  # Это префикс для названий файлов!
        Lx=20.0,  # Ваша новая ширина поля
        Nx=80,  # Ваше количество ячеек (если dx остался прежним)
    )
    plot_ismd_results_2d(
        times=res_2d_t2_t["times"],
        populations=res_2d_t2_t["populations"],
        max_coords=res_2d_t2_t["max"],
        snapshots=res_2d_t2_t["snapshots"],
        sim_type="type2_t_20x20",  # Это префикс для названий файлов!
        Lx=20.0,  # Ваша новая ширина поля
        Nx=80,  # Ваше количество ячеек (если dx остался прежним)
    )
    plot_ismd_results_2d(
        times=res_2d_t1_h["times"],
        populations=res_2d_t1_h["populations"],
        max_coords=res_2d_t1_h["max"],
        snapshots=res_2d_t1_h["snapshots"],
        sim_type="type1_h_10x10",  # Это префикс для названий файлов!
        Lx=10.0,  # Ваша новая ширина поля
        Nx=80  # Ваше количество ячеек (если dx остался прежним)
    )
    plot_ismd_results_2d(
        times=res_2d_t1_t["times"],
        populations=res_2d_t1_t["populations"],
        max_coords=res_2d_t1_t["max"],
        snapshots=res_2d_t1_t["snapshots"],
        sim_type="type1_t_10x10",  # Это префикс для названий файлов!
        Lx=10.0,  # Ваша новая ширина поля
        Nx=80  # Ваше количество ячеек (если dx остался прежним)
    )
    #3D plots
    # _plot_max_distance(
    #     res_3d_t1["times"],
    #     res_3d_t1["max"],
    #     res_3d_t1["reg_max"],
    #     "3D type 1 front max distance vs time",
    #     "Max distance from center",
    #     OUTPUT_DIR / "3d_max_distance_t1.png",
    # )
    # _plot_max_distance(
    #     res_3d_t2["times"],
    #     res_3d_t2["max"],
    #     res_3d_t2["reg_max"],
    #     "3D type 2 front max distance vs time",
    #     "Max distance from center",
    #     OUTPUT_DIR / "3d_max_distance_t2.png",
    # )
    # Логарифмические графики отклонений для Type 1
    # _plot_log_difference_left_right(
    #     res_1d_t1["times"], res_1d_t1["left"], res_1d_t1["right"],
    #     "1D Type 1: Left vs Right Asymmetry (Log Scale)",
    #     OUTPUT_DIR / "1d_ismd_log_diff_lr_t1.png",
    # )
    # _plot_log_deviations_from_fit(
    #     res_1d_t1["times"], res_1d_t1["max"], res_1d_t1["reg_max"],
    #     "1D Type 1: Deviation from Linear Fit (Log Scale)",
    #     OUTPUT_DIR / "1d_ismd_log_dev_fit_t1.png",
    # )
    #
    # # Логарифмические графики отклонений для Type 2
    # _plot_log_difference_left_right(
    #     res_1d_t2["times"], res_1d_t2["left"], res_1d_t2["right"],
    #     "1D Type 2: Left vs Right Asymmetry (Log Scale)",
    #     OUTPUT_DIR / "1d_ismd_log_diff_lr_t2.png",
    # )
    # _plot_log_deviations_from_fit(
    #     res_1d_t2["times"], res_1d_t2["max"], res_1d_t2["reg_max"],
    #     "1D Type 2: Deviation from Linear Fit (Log Scale)",
    #     OUTPUT_DIR / "1d_ismd_log_dev_fit_t2.png",
    # )
    # Console summary

    print("\nISMD Front speed slopes (units distance per unit time):")
    if res_1d_t1_h["reg_max"] is not None:
        print(f"  1D t1 (Half-Normal): {res_1d_t1_h['reg_max'].slope:.6f}")
    if res_1d_t1_t["reg_max"] is not None:
        print(f"  1D t1 (Top-Hat):     {res_1d_t1_t['reg_max'].slope:.6f}")
    if res_1d_t2_h["reg_max"] is not None:
        print(f"  1D t2 (Half-Normal): {res_1d_t2_h['reg_max'].slope:.6f}")
    if res_1d_t2_t["reg_max"] is not None:
        print(f"  1D t2 (Top-Hat):     {res_1d_t2_t['reg_max'].slope:.6f}")

    if res_2d_t1_h["reg_max"] is not None:
        print(f"  2D t1 (Half-Normal): {res_2d_t1_h['reg_max'].slope:.6f}")
    if res_2d_t1_t["reg_max"] is not None:
        print(f"  2D t1 (Top-Hat):     {res_2d_t1_t['reg_max'].slope:.6f}")
    if res_2d_t2_h["reg_max"] is not None:
        print(f"  2D t2 (Half-Normal): {res_2d_t2_h['reg_max'].slope:.6f}")
    if res_2d_t2_t["reg_max"] is not None:
        print(f"  2D t2 (Top-Hat):     {res_2d_t2_t['reg_max'].slope:.6f}")

    # print("\nISMD Front speed slopes (units distance per unit time):")
    # if res_1d_t1_h["reg_max"] is not None:
    #      print(f"  1D (overall t1): {res_1d_t1_h['reg_max'].slope:.6f}")
    # if res_1d_t1_t["reg_max"] is not None:
    #     print(f"  1D (overall t1): {res_1d_t1_t['reg_max'].slope:.6f}")
    # if res_1d_t2_h["reg_max"] is not None:
    #     print(f"  1D (overall t2): {res_1d_t2_h['reg_max'].slope:.6f}")
    # if res_1d_t2_t["reg_max"] is not None:
    #     print(f"  1D (overall t2): {res_1d_t2_t['reg_max'].slope:.6f}")
    # if res_2d_t1_h["reg_max"] is not None:
    #     print(f"  2D t1:           {res_2d_t1_h['reg_max'].slope:.6f}")
    # if res_2d_t1_t["reg_max"] is not None:
    #     print(f"  2D t1:           {res_2d_t1_t['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")
    # if res_2d_t2_h["reg_max"] is not None:
    #     print(f"  2D t2:           {res_2d_t2_h['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")
    # if res_2d_t2_t["reg_max"] is not None:
    #     print(f"  2D t2:           {res_2d_t2_t['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")
    # if res_3d_t1["reg_max"] is not None:
    #     print(f"  3D t1:           {res_3d_t1['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")
    # if res_3d_t2["reg_max"] is not None:
    #     print(f"  3D t2:           {res_3d_t2['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()