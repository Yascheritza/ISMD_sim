import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SSA.front_analysis import (
    article_front_speeds,
    front_radii_at_levels,
    linear_spreading_speed_3d,
    radial_profile_3d,
)
from SSA.model_kernels import gaussian_kernel_3d, tophat_kernel_3d


OUTPUT_DIR = Path(__file__).resolve().parent / "front_speed_results"
ARTICLE_LEVELS = (0.25, 0.5, 0.75)


def _model_parameters(model: str, birth_sigma: float | None, competition_sigma: float | None):
    if model == "type1":
        kernel_name = "tophat"
        make_kernel = tophat_kernel_3d
        default_birth_sigma, default_competition_sigma = 0.1, 1.0
        default_fit_start = 32.0
    elif model == "type2":
        kernel_name = "gaussian"
        make_kernel = gaussian_kernel_3d
        default_birth_sigma, default_competition_sigma = 1.0, 0.1
        default_fit_start = 24.0
    else:
        raise ValueError("model must be 'type1' or 'type2'")

    return (
        kernel_name,
        make_kernel,
        default_birth_sigma if birth_sigma is None else birth_sigma,
        default_competition_sigma if competition_sigma is None else competition_sigma,
        default_fit_start,
    )


def _initial_density_3d(
    area_size: Sequence[float],
    cell_counts: Sequence[int],
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    coordinates = [
        (np.arange(count, dtype=np.float64) + 0.5) * length / count - length / 2.0
        for length, count in zip(area_size, cell_counts)
    ]
    zz, yy, xx = np.meshgrid(coordinates[2], coordinates[1], coordinates[0], indexing="ij")
    return amplitude * np.exp(-(xx * xx + yy * yy + zz * zz) / (2.0 * sigma * sigma)).ravel()


def _front_regression(times, fronts, fit_start_time: float, level_index: int):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(fronts, dtype=np.float64)[:, level_index]
    selected = (times >= fit_start_time) & np.isfinite(values)
    if np.count_nonzero(selected) < 2:
        return None
    return linregress(times[selected], values[selected])


def measure_front_speed_3d(
    model: str,
    time_step: float = 1.0,
    area_size: Sequence[float] = (80.0, 80.0, 80.0),
    cell_counts: Sequence[int] = (10, 10, 10),
    dt: float = 0.05,
    max_time: float = 40.0,
    fit_start_time: float | None = None,
    c_plus: float = 1.0,
    c_minus: float = 1.0,
    mortality: float = 0.01,
    birth_sigma: float | None = None,
    competition_sigma: float | None = None,
    initial_amplitude: float = 1.0,
    initial_sigma: float = 1.0,
    levels: Sequence[float] = ARTICLE_LEVELS,
):
    """Запускает 3D ISMD/KSA/DP и измеряет радиальную скорость фронта."""
    if dt <= 0.0 or time_step <= 0.0 or max_time <= 0.0:
        raise ValueError("time steps and max_time must be positive")
    if len(area_size) != 3 or len(cell_counts) != 3:
        raise ValueError("3D simulation needs three area sizes and cell counts")

    kernel_name, make_kernel, sigma_plus, sigma_minus, article_fit_start = _model_parameters(
        model, birth_sigma, competition_sigma
    )
    if fit_start_time is None:
        fit_start_time = article_fit_start

    birth_x, birth_y = make_kernel(sigma_plus, c_plus)
    death_x, death_y = make_kernel(sigma_minus, c_minus)
    n_init = _initial_density_3d(area_size, cell_counts, initial_amplitude, initial_sigma)

    # В статье начальное состояние пуассоновское: g0=1, поэтому u0(x,y)=n0(x)n0(y).
    from SSA.ismd_sim import make_ismd_state

    simulation = make_ismd_state(
        ndim=3,
        area_size=area_size,
        cell_counts=cell_counts,
        m=mortality,
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        periodic=False,
        n_init=n_init,
        u_init=np.outer(n_init, n_init),
    )

    radii, previous_profile = radial_profile_3d(n_init, area_size, cell_counts)
    front = front_radii_at_levels(radii, previous_profile, levels)
    times = [0.0]
    fronts = [front]
    local_speeds = [np.full(len(levels), np.nan, dtype=np.float64)]
    profiles = [previous_profile]
    slices = [n_init.reshape(tuple(reversed(cell_counts)))[cell_counts[2] // 2]]
    populations = [float(n_init.sum() * np.prod(np.asarray(area_size) / np.asarray(cell_counts)))]

    record_time = time_step
    previous_time = 0.0
    stop_reason = "max_time"
    invalid_density_min = None
    interaction_range = float(max(birth_x[-2], death_x[-2]))
    radial_limit = min(area_size) / 2.0
    total_steps = int(np.ceil(max_time / dt))

    for _ in range(total_steps):
        simulation.step(dt)
        if simulation.time + 1e-12 < record_time and simulation.time + 1e-12 < max_time:
            continue

        density = simulation.get_density()
        density_min = float(np.min(density))
        if not np.all(np.isfinite(density)) or density_min < 0.0:
            stop_reason = "invalid_density"
            invalid_density_min = density_min
            break
        _, profile = radial_profile_3d(density, area_size, cell_counts)
        front = front_radii_at_levels(radii, profile, levels)
        speed = article_front_speeds(
            radii, previous_profile, profile, simulation.time - previous_time, front
        )

        times.append(float(simulation.time))
        fronts.append(front)
        local_speeds.append(speed)
        profiles.append(profile)
        slices.append(density.reshape(tuple(reversed(cell_counts)))[cell_counts[2] // 2])
        populations.append(
            float(density.sum() * np.prod(np.asarray(area_size) / np.asarray(cell_counts)))
        )
        previous_profile = profile
        previous_time = float(simulation.time)
        record_time += time_step

        finite_front = front[np.isfinite(front)]
        if finite_front.size and finite_front.max() + interaction_range >= radial_limit:
            stop_reason = "front_reached_dirichlet_margin"
            break
        if simulation.time + 1e-12 >= max_time:
            break

    middle_level = int(np.argmin(np.abs(np.asarray(levels, dtype=np.float64) - 0.5)))
    regression = _front_regression(times, fronts, fit_start_time, middle_level)
    return {
        "model": model,
        "kernel": kernel_name,
        "times": np.asarray(times),
        "radii": radii,
        "fronts": np.asarray(fronts),
        "local_speeds": np.asarray(local_speeds),
        "profiles": np.asarray(profiles),
        "slices": np.asarray(slices),
        "populations": np.asarray(populations),
        "levels": np.asarray(levels),
        "reg_front": regression,
        "fit_start_time": fit_start_time,
        "area_size": tuple(area_size),
        "cell_counts": tuple(cell_counts),
        "birth_sigma": sigma_plus,
        "competition_sigma": sigma_minus,
        "c_plus": c_plus,
        "c_minus": c_minus,
        "mortality": mortality,
        "stop_reason": stop_reason,
        "invalid_density_min": invalid_density_min,
    }


def measure_front_speed_3d_type1(**kwargs):
    return measure_front_speed_3d("type1", **kwargs)


def measure_front_speed_3d_type2(**kwargs):
    return measure_front_speed_3d("type2", **kwargs)


def plot_3d_front_results(result, output: Path) -> None:
    """Строит траектории фронта, локальную скорость, профили и центральные срезы."""
    output.parent.mkdir(parents=True, exist_ok=True)
    times = result["times"]
    levels = result["levels"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")

    for index, level in enumerate(levels):
        axes[0, 0].plot(times, result["fronts"][:, index], marker=".", label=f"n/max(n)={level:g}")
        axes[0, 1].plot(times, result["local_speeds"][:, index], marker=".", label=f"level={level:g}")
    regression = result["reg_front"]
    if regression is not None:
        selected = times >= result["fit_start_time"]
        axes[0, 0].plot(
            times[selected],
            regression.intercept + regression.slope * times[selected],
            "k--",
            label=f"fit: v={regression.slope:.4g}",
        )
    axes[0, 0].set(title="Положение сферического фронта", xlabel="t", ylabel="r")
    axes[0, 1].set(title=r"Локальная скорость $-\partial_t n/\partial_r n$", xlabel="t", ylabel="v")

    selected = np.unique(np.linspace(0, len(times) - 1, min(5, len(times)), dtype=int))
    for index in selected:
        axes[1, 0].plot(result["radii"], result["profiles"][index], label=f"t={times[index]:g}")
    axes[1, 0].set(title="Сферически-усреднённая плотность", xlabel="r", ylabel="n(r)")

    image = axes[1, 1].imshow(
        result["slices"][-1],
        origin="lower",
        extent=(-result["area_size"][0] / 2.0, result["area_size"][0] / 2.0,
                -result["area_size"][1] / 2.0, result["area_size"][1] / 2.0),
    )
    axes[1, 1].set(title=f"Центральный z-срез, t={times[-1]:g}", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=axes[1, 1], label="n")
    for axis in axes.flat[:3]:
        axis.grid(True, linestyle=":", linewidth=0.7)
        axis.legend()
    fig.suptitle(f"3D ISMD, {result['model']}")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _run(model: str, args) -> None:
    _, _, sigma_plus, sigma_minus, _ = _model_parameters(
        model, args.birth_sigma, args.competition_sigma
    )
    spatial_step = args.length / args.cells
    if spatial_step > min(sigma_plus, sigma_minus):
        print(
            f"warning: grid step {spatial_step:g} does not resolve "
            f"the shortest kernel scale {min(sigma_plus, sigma_minus):g}"
        )
    result = measure_front_speed_3d(
        model,
        time_step=args.record_interval,
        area_size=(args.length,) * 3,
        cell_counts=(args.cells,) * 3,
        dt=args.dt,
        max_time=args.max_time,
        fit_start_time=args.fit_start_time,
        c_plus=args.c_plus,
        c_minus=args.c_minus,
        mortality=args.mortality,
        birth_sigma=args.birth_sigma,
        competition_sigma=args.competition_sigma,
    )
    plot_3d_front_results(result, args.output_dir / f"3d_ismd_{model}.png")

    regression = result["reg_front"]
    finite_speeds = result["local_speeds"][np.isfinite(result["local_speeds"])]
    print(f"{model}: stop={result['stop_reason']}")
    if regression is not None:
        print(f"{model}: trajectory speed={regression.slope:.6f}")
    if finite_speeds.size:
        print(f"{model}: mean local speed={finite_speeds.mean():.6f}")
    linear_speed, optimal_lambda = linear_spreading_speed_3d(
        result["kernel"], args.c_plus, args.mortality, result["birth_sigma"]
    )
    print(f"{model}: linear MF estimate={linear_speed:.6f}, lambda={optimal_lambda:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Расчёт скорости 3D-фронта ISMD по определению из статьи.")
    parser.add_argument("--model", choices=("type1", "type2", "both"), default="both")
    parser.add_argument("--length", type=float, default=80.0)
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--record-interval", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--fit-start-time", type=float)
    parser.add_argument("--c-plus", type=float, default=1.0)
    parser.add_argument("--c-minus", type=float, default=1.0)
    parser.add_argument("--mortality", type=float, default=0.01)
    parser.add_argument("--birth-sigma", type=float)
    parser.add_argument("--competition-sigma", type=float)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    models = ("type1", "type2") if args.model == "both" else (args.model,)
    for model in models:
        _run(model, args)


if __name__ == "__main__":
    main()
