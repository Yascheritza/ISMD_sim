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

from SSA import (
    make_ssa_state_1d,
    make_ssa_state_2d,
    make_ssa_state_3d,
    get_all_particle_coords,
)
from tests.test_equal_kernels_nd import (
    tophat_equal_tables_1d,
    half_normal_equal_tables_1d,
    half_normal_equal_tables_2d,
    half_normal_equal_tables_3d, tophat_equal_tables_3d, tophat_equal_tables_2d,
)


TIME_STEP = 0.05
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


def measure_front_speed_1d(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.01
    dd = 1.0
    cutoff = 5.0
    L = 80
    cells = 6400

    result_birth = half_normal_equal_tables_1d(cutoff, sigma=1.0)
    result_death = half_normal_equal_tables_1d(cutoff, sigma=0.1)
    m_d_x = result_death[2]
    m_d_y = result_death[3]
    m_b_x = result_birth[0]
    m_b_y = result_birth[1]
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=m_b_x,
        birth_y=m_b_y,
        death_x=m_d_x,
        death_y=m_d_y,
        cutoffs=np.array([[cutoff]]),
        seed=101,
        cell_capacity=4096,
        is_periodic=False,
    )

    center = 0.5 * L
    sim.spawn_particle(0, center)

    times: list[float] = []
    max_radii: list[float] = []
    left_radii: list[float] = []
    right_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        dx = xs - center
        max_radius = float(np.abs(dx).max())
        left_dist = float(center - xs.min())
        right_dist = float(xs.max() - center)

        times.append(sim.current_time())
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        boundary_reached = xs.min() <= cutoff or (L - xs.max()) <= cutoff
        if boundary_reached:
            times.pop()
            max_radii.pop()
            left_radii.pop()
            right_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)
    left_arr = np.asarray(left_radii, dtype=np.float64)
    right_arr = np.asarray(right_radii, dtype=np.float64)





    return {
        "times": time_arr,
        "max": max_arr,
        "left": left_arr,
        "right": right_arr,
        "reg_max": _regression(time_arr, max_arr),
        "reg_left": _regression(time_arr, left_arr),
        "reg_right": _regression(time_arr, right_arr),
        "cutoff": cutoff,
        "length": L,
        "center": center,
    }


def measure_front_speed_1d_t1_t(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.01
    dd = 1.0
    cutoff = 5.0
    L = 40
    cells = 3200

    result_birth = tophat_equal_tables_1d(cutoff, sigma=0.1)
    result_death = tophat_equal_tables_1d(cutoff, sigma=1.0)
    m_d_x = result_death[2]
    m_d_y = result_death[3]
    m_b_x = result_birth[0]
    m_b_y = result_birth[1]
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=np.array([m_b_x]),
        birth_y=np.array([m_b_y]),
        death_x=np.array([[m_d_x]]),
        death_y=np.array([[m_d_y]]),
        cutoffs=np.array([[cutoff]]),
        seed=101,
        cell_capacity=4096,
        is_periodic=False,
    )

    center = 0.5 * L
    sim.spawn_particle(0, center)

    times: list[float] = []
    max_radii: list[float] = []
    left_radii: list[float] = []
    right_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        dx = xs - center
        max_radius = float(np.abs(dx).max())
        left_dist = float(center - xs.min())
        right_dist = float(xs.max() - center)

        times.append(sim.current_time())
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        boundary_reached = xs.min() <= cutoff or (L - xs.max()) <= cutoff
        if boundary_reached:
            times.pop()
            max_radii.pop()
            left_radii.pop()
            right_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)
    left_arr = np.asarray(left_radii, dtype=np.float64)
    right_arr = np.asarray(right_radii, dtype=np.float64)





    return {
        "times": time_arr,
        "max": max_arr,
        "left": left_arr,
        "right": right_arr,
        "reg_max": _regression(time_arr, max_arr),
        "reg_left": _regression(time_arr, left_arr),
        "reg_right": _regression(time_arr, right_arr),
        "cutoff": cutoff,
        "length": L,
        "center": center,
    }


def measure_front_speed_1d_t1_h(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.01
    dd = 1.0
    cutoff = 5.0
    L = 40
    cells = 3200

    result_birth = half_normal_equal_tables_1d(cutoff, sigma=0.1)
    result_death = half_normal_equal_tables_1d(cutoff, sigma=1.0)
    m_d_x = result_death[2]
    m_d_y = result_death[3]
    m_b_x = result_birth[0]
    m_b_y = result_birth[1]
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=np.array([m_b_x]),
        birth_y=np.array([m_b_y]),
        death_x=np.array([[m_d_x]]),
        death_y=np.array([[m_d_y]]),
        cutoffs=np.array([[cutoff]]),
        seed=101,
        cell_capacity=4096,
        is_periodic=False,
    )

    center = 0.5 * L
    sim.spawn_particle(0, center)

    times: list[float] = []
    max_radii: list[float] = []
    left_radii: list[float] = []
    right_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        dx = xs - center
        max_radius = float(np.abs(dx).max())
        left_dist = float(center - xs.min())
        right_dist = float(xs.max() - center)

        times.append(sim.current_time())
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        boundary_reached = xs.min() <= cutoff or (L - xs.max()) <= cutoff
        if boundary_reached:
            times.pop()
            max_radii.pop()
            left_radii.pop()
            right_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)
    left_arr = np.asarray(left_radii, dtype=np.float64)
    right_arr = np.asarray(right_radii, dtype=np.float64)





    return {
        "times": time_arr,
        "max": max_arr,
        "left": left_arr,
        "right": right_arr,
        "reg_max": _regression(time_arr, max_arr),
        "reg_left": _regression(time_arr, left_arr),
        "reg_right": _regression(time_arr, right_arr),
        "cutoff": cutoff,
        "length": L,
        "center": center,
    }


def measure_front_speed_1d_t2_t(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.01
    dd = 1.0
    cutoff = 5.0
    L = 40
    cells = 3200

    result_birth = tophat_equal_tables_1d(cutoff, sigma=1.0)
    result_death = tophat_equal_tables_1d(cutoff, sigma=0.1)
    m_d_x = result_death[2]
    m_d_y = result_death[3]
    m_b_x = result_birth[0]
    m_b_y = result_birth[1]
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=np.array([m_b_x]),
        birth_y=np.array([m_b_y]),
        death_x=np.array([[m_d_x]]),
        death_y=np.array([[m_d_y]]),
        cutoffs=np.array([[cutoff]]),
        seed=101,
        cell_capacity=4096,
        is_periodic=False,
    )

    center = 0.5 * L
    sim.spawn_particle(0, center)

    times: list[float] = []
    max_radii: list[float] = []
    left_radii: list[float] = []
    right_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        dx = xs - center
        max_radius = float(np.abs(dx).max())
        left_dist = float(center - xs.min())
        right_dist = float(xs.max() - center)

        times.append(sim.current_time())
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        boundary_reached = xs.min() <= cutoff or (L - xs.max()) <= cutoff
        if boundary_reached:
            times.pop()
            max_radii.pop()
            left_radii.pop()
            right_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)
    left_arr = np.asarray(left_radii, dtype=np.float64)
    right_arr = np.asarray(right_radii, dtype=np.float64)





    return {
        "times": time_arr,
        "max": max_arr,
        "left": left_arr,
        "right": right_arr,
        "reg_max": _regression(time_arr, max_arr),
        "reg_left": _regression(time_arr, left_arr),
        "reg_right": _regression(time_arr, right_arr),
        "cutoff": cutoff,
        "length": L,
        "center": center,
    }


def measure_front_speed_1d_t2_h(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.01
    dd = 1.0
    cutoff = 5.0
    L = 40
    cells = 3200

    result_birth = half_normal_equal_tables_1d(cutoff, sigma=1.0)
    result_death = half_normal_equal_tables_1d(cutoff, sigma=0.1)
    m_d_x = result_death[2]
    m_d_y = result_death[3]
    m_b_x = result_birth[0]
    m_b_y = result_birth[1]
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=np.array([m_b_x]),
        birth_y=np.array([m_b_y]),
        death_x=np.array([[m_d_x]]),
        death_y=np.array([[m_d_y]]),
        cutoffs=np.array([[cutoff]]),
        seed=101,
        cell_capacity=4096,
        is_periodic=False,
    )

    center = 0.5 * L
    sim.spawn_particle(0, center)

    times: list[float] = []
    max_radii: list[float] = []
    left_radii: list[float] = []
    right_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        dx = xs - center
        max_radius = float(np.abs(dx).max())
        left_dist = float(center - xs.min())
        right_dist = float(xs.max() - center)

        times.append(sim.current_time())
        max_radii.append(max_radius)
        left_radii.append(left_dist)
        right_radii.append(right_dist)

        boundary_reached = xs.min() <= cutoff or (L - xs.max()) <= cutoff
        if boundary_reached:
            times.pop()
            max_radii.pop()
            left_radii.pop()
            right_radii.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)
    left_arr = np.asarray(left_radii, dtype=np.float64)
    right_arr = np.asarray(right_radii, dtype=np.float64)





    return {
        "times": time_arr,
        "max": max_arr,
        "left": left_arr,
        "right": right_arr,
        "reg_max": _regression(time_arr, max_arr),
        "reg_left": _regression(time_arr, left_arr),
        "reg_right": _regression(time_arr, right_arr),
        "cutoff": cutoff,
        "length": L,
        "center": center,
    }



def measure_front_speed_2d(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.0
    dd = 0.1
    cutoff = 5.0
    Lx, Ly = 100.0, 100.0
    Nx, Ny = 100, 100

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_2d(cutoff)
    sim = make_ssa_state_2d(
        M=1,
        area_len=np.array([Lx, Ly]),
        cell_count=np.array([Nx, Ny]),
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=202,
        is_periodic=False,
    )

    cx, cy = 0.5 * Lx, 0.5 * Ly
    sim.spawn_particle(0, cx, cy)

    times: list[float] = []
    max_radii: list[float] = []
    angular_bins_history: list[np.ndarray] = []
    polygons: list[np.ndarray] = []

    num_bins = int(360 / DEGREE_BIN)
    angles = np.deg2rad(np.arange(0, 360, DEGREE_BIN))
    angles_ext = np.append(angles, angles[0])

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        ys = np.array([sim.positions[i, 1] for i in range(total)], dtype=np.float64)
        dx = xs - cx
        dy = ys - cy
        radii = np.hypot(dx, dy)
        max_radius = float(radii.max())

        phi = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        indices = np.floor(phi / DEGREE_BIN).astype(int) % num_bins
        bin_max = np.zeros(num_bins, dtype=np.float64)
        for idx, r in zip(indices, radii):
            if r > bin_max[idx]:
                bin_max[idx] = r

        radii_ext = np.append(bin_max, bin_max[0])
        polygon = np.column_stack((cx + radii_ext * np.cos(angles_ext), cy + radii_ext * np.sin(angles_ext)))

        times.append(sim.current_time())
        max_radii.append(max_radius)
        angular_bins_history.append(bin_max)
        polygons.append(polygon)

        boundary_reached = (
            xs.min() <= cutoff
            or (Lx - xs.max()) <= cutoff
            or ys.min() <= cutoff
            or (Ly - ys.max()) <= cutoff
        )
        if boundary_reached:
            times.pop()
            max_radii.pop()
            angular_bins_history.pop()
            polygons.pop()
            break

    time_arr = np.asarray(times, dtype=np.float64)
    max_arr = np.asarray(max_radii, dtype=np.float64)

    return {
        "times": time_arr,
        "max": max_arr,
        "angular_bins": angular_bins_history,
        "polygons": polygons,
        "reg_max": _regression(time_arr, max_arr),
        "cutoff": cutoff,
        "extent": (0.0, Lx, 0.0, Ly),
    }


def measure_front_speed_3d(time_step: float = TIME_STEP):
    b = 1.0
    d = 0.0
    dd = 0.1
    cutoff = 5.0
    Lx, Ly, Lz = 50.0, 50.0, 50.0
    Nx, Ny, Nz = 50, 50, 50

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_3d(cutoff)
    sim = make_ssa_state_3d(
        M=1,
        area_len=np.array([Lx, Ly, Lz]),
        cell_count=np.array([Nx, Ny, Nz]),
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=303,
        cell_capacity=1024,
        is_periodic=False,
    )

    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    sim.spawn_particle(0, cx, cy, cz)

    times: list[float] = []
    max_radii: list[float] = []

    while True:
        sim.run_until_time(time_step)
        total = sim.current_population()
        if total <= 0:
            break

        xs = np.array([sim.positions[i, 0] for i in range(total)], dtype=np.float64)
        ys = np.array([sim.positions[i, 1] for i in range(total)], dtype=np.float64)
        zs = np.array([sim.positions[i, 2] for i in range(total)], dtype=np.float64)
        dx = xs - cx
        dy = ys - cy
        dz = zs - cz
        radii = np.sqrt(dx * dx + dy * dy + dz * dz)
        max_radius = float(radii.max())

        times.append(sim.current_time())
        max_radii.append(max_radius)

        boundary_reached = (
            xs.min() <= cutoff
            or (Lx - xs.max()) <= cutoff
            or ys.min() <= cutoff
            or (Ly - ys.max()) <= cutoff
            or zs.min() <= cutoff
            or (Lz - zs.max()) <= cutoff
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


def main():

    res_1d_t1_t = measure_front_speed_1d_t1_t()
    res_1d_t1_h = measure_front_speed_1d_t1_h()
    res_1d_t2_t = measure_front_speed_1d_t2_t()
    res_1d_t2_h = measure_front_speed_1d_t2_h()
    # res_2d = measure_front_speed_2d()
    # res_3d = measure_front_speed_3d()

    # 1D plots
    _plot_max_distance(
        res_1d_t1_t["times"],
        res_1d_t1_t["max"],
        res_1d_t1_t["reg_max"],
        "1D t1 t front max distance vs time",
        "Max distance from center",
        OUTPUT_DIR / "1d_t1_t_ssa_max_distance.png",
    )
    _plot_max_distance(
        res_1d_t1_h["times"],
        res_1d_t1_h["max"],
        res_1d_t1_h["reg_max"],
        "1D t1 h front max distance vs time",
        "Max distance from center",
        OUTPUT_DIR / "1d_t1_h_ssa_max_distance.png",
    )
    _plot_max_distance(
        res_1d_t2_t["times"],
        res_1d_t2_t["max"],
        res_1d_t2_t["reg_max"],
        "1D t2 t front max distance vs time",
        "Max distance from center",
        OUTPUT_DIR / "1d_t2_t_ssa_max_distance.png",
    )
    _plot_max_distance(
        res_1d_t2_h["times"],
        res_1d_t2_h["max"],
        res_1d_t2_h["reg_max"],
        "1D t2 h front max distance vs time",
        "Max distance from center",
        OUTPUT_DIR / "1d_t2_h_ssa_max_distance.png",
    )
    # _plot_left_right(
    #     res_1d["times"],
    #     res_1d["left"],
    #     res_1d["right"],
    #     res_1d["reg_left"],
    #     res_1d["reg_right"],
    #     OUTPUT_DIR / "1d_left_right_distance.png",
    # )

    # 2D plots
    # _plot_max_distance(
    #     res_2d["times"],
    #     res_2d["max"],
    #     res_2d["reg_max"],
    #     "2D front max distance vs time",
    #     "Max distance from center",
    #     OUTPUT_DIR / "2d_max_distance.png",
    # )
    # _plot_2d_polygons(
    #     res_2d["polygons"],
    #     res_2d["times"],
    #     res_2d["extent"],
    #     OUTPUT_DIR / "2d_border_polygons.png",
    # )

    # 3D plots
    # _plot_max_distance(
    #     res_3d["times"],
    #     res_3d["max"],
    #     res_3d["reg_max"],
    #     "3D front max distance vs time",
    #     "Max distance from center",
    #     OUTPUT_DIR / "3d_max_distance.png",
    # )

    # Console summary
    print("Front speed slopes (units distance per unit time):")
    if res_1d_t1_t["reg_max"] is not None:
        print(f"  1D t1 t (overall): {res_1d_t1_t['reg_max'].slope:.6f}")
    if res_1d_t1_h["reg_max"] is not None:
        print(f"  1D t1 h (overall): {res_1d_t1_h['reg_max'].slope:.6f}")
    if res_1d_t2_t["reg_max"] is not None:
        print(f"  1D t2 t (overall): {res_1d_t2_t['reg_max'].slope:.6f}")
    if res_1d_t2_h["reg_max"] is not None:
        print(f"  1D t2 h (overall): {res_1d_t2_h['reg_max'].slope:.6f}")
    # if res_1d["reg_left"] is not None:
    #     print(f"  1D (left):    {res_1d['reg_left'].slope:.6f}")
    # if res_1d["reg_right"] is not None:
    #     print(f"  1D (right):   {res_1d['reg_right'].slope:.6f}")
    # if res_2d["reg_max"] is not None:
    #     print(f"  2D:           {res_2d['reg_max'].slope:.6f}")
    # if res_3d["reg_max"] is not None:
    #     print(f"  3D:           {res_3d['reg_max'].slope:.6f}")
    # print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()