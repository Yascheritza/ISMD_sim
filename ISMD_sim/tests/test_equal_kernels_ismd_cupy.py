import math
import numpy as np
from scipy.stats import halfnorm, rayleigh, maxwell

from SSA import (
    make_ssa_state_1d,
    make_ssa_state_2d,
    make_ssa_state_3d,
)

STD_SIGMA = 1.0
STD_CUTOFF = 5.0
STD_KERNEL_GRID = 512
STD_SAMPLE_POINTS = 192
STD_B = 1.0
STD_D = 0.0
STD_DD = 0.1
STD_EQ_POPULATION = 10_000
EQ_TOLERANCE = 0.05


def half_normal_equal_tables_1d(
    cutoff: float = STD_CUTOFF,
    sigma: float = STD_SIGMA,
    grid: int = STD_KERNEL_GRID,
    samples: int = STD_SAMPLE_POINTS,
):
    dist = halfnorm(scale=sigma)
    mass = dist.cdf(cutoff)
    if mass <= 0.0:
        raise ValueError("halfnorm mass at cutoff is zero; increase cutoff or adjust sigma.")
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)
    pdf_trunc = dist.pdf(r) / mass
    w = 0.5 * pdf_trunc

    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    upper = mass * (1.0 - 0.5 / samples)
    if upper <= 1e-10:
        upper = mass
    ppf_args = np.clip(quantiles * mass, 1e-12, upper)
    radii = dist.ppf(ppf_args)
    radii = np.clip(radii, 0.0, cutoff)

    return quantiles, radii, r, w


def tophat_equal_tables_1d(
        cutoff: float = STD_CUTOFF,
        sigma: float = STD_SIGMA,
        grid: int = STD_KERNEL_GRID,
        samples: int = STD_SAMPLE_POINTS,
):
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)
    density_value = 1.0 / (2.0 * sigma)
    w = np.zeros_like(r)
    w[r <= sigma] = density_value
    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    radii = quantiles * sigma
    radii = np.clip(radii, 0.0, cutoff)

    return quantiles, radii, r, w

def half_normal_equal_tables_2d(
    cutoff: float = STD_CUTOFF,
    sigma: float = STD_SIGMA,
    grid: int = STD_KERNEL_GRID,
    samples: int = STD_SAMPLE_POINTS,
):
    dist = rayleigh(scale=sigma)
    mass = dist.cdf(cutoff)
    if mass <= 0.0:
        raise ValueError("rayleigh mass at cutoff is zero; increase cutoff or adjust sigma.")
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)
    pdf_trunc = dist.pdf(r) / mass
    w = np.empty_like(pdf_trunc)
    if w.size:
        w[0] = (1.0 / (2.0 * math.pi * sigma * sigma)) / mass
        w[1:] = pdf_trunc[1:] / (2.0 * math.pi * r[1:])

    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    upper = mass * (1.0 - 0.5 / samples)
    if upper <= 1e-10:
        upper = mass
    ppf_args = np.clip(quantiles * mass, 1e-12, upper)
    radii = dist.ppf(ppf_args)
    radii = np.clip(radii, 0.0, cutoff)

    # Форматирование под архитектуру ISMD (плоские 1D массивы)
    return quantiles, radii, r, w

def tophat_equal_tables_2d(
        cutoff: float = STD_CUTOFF,  # Не забудьте убедиться, что константы импортированы
        sigma: float = STD_SIGMA,
        grid: int = STD_KERNEL_GRID,
        samples: int = STD_SAMPLE_POINTS,
):
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)

    # 1. Плотность взаимодействия (константа внутри круга, 0 снаружи)
    area = math.pi * (sigma ** 2)
    density_value = 1.0 / area

    w = np.zeros_like(r)
    w[r <= sigma] = density_value

    # 2. Вычисление радиусов для спавна потомков
    # Берем квантили от 0 до 1
    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples

    # Для равномерного распределения в 2D (площадь круга) извлекаем квадратный корень
    radii = sigma * np.sqrt(quantiles)
    radii = np.clip(radii, 0.0, cutoff)

    # Форматирование под архитектуру симулятора
    # Форматирование под архитектуру ISMD (плоские 1D массивы)
    return quantiles, radii, r, w

def half_normal_equal_tables_3d(
    cutoff: float = STD_CUTOFF,
    sigma: float = STD_SIGMA,
    grid: int = STD_KERNEL_GRID,
    samples: int = STD_SAMPLE_POINTS,
):
    dist = maxwell(scale=sigma)
    mass = dist.cdf(cutoff)
    if mass <= 0.0:
        raise ValueError("maxwell mass at cutoff is zero; increase cutoff or adjust sigma.")
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)
    pdf_trunc = dist.pdf(r) / mass
    w = np.empty_like(pdf_trunc)
    if w.size:
        w[0] = (math.sqrt(2.0 / math.pi) / (4.0 * math.pi * sigma ** 3)) / mass
        if w.size > 1:
            denom = 4.0 * math.pi * r[1:] * r[1:]
            w[1:] = pdf_trunc[1:] / denom

    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    upper = mass * (1.0 - 0.5 / samples)
    if upper <= 1e-10:
        upper = mass
    ppf_args = np.clip(quantiles * mass, 1e-12, upper)
    radii = dist.ppf(ppf_args)
    radii = np.clip(radii, 0.0, cutoff)

    # Форматирование под архитектуру ISMD (плоские 1D массивы)
    return quantiles, radii, r, w

def tophat_equal_tables_3d(
        cutoff: float = STD_CUTOFF,  # Убедитесь, что эти константы импортированы
        sigma: float = STD_SIGMA,
        grid: int = STD_KERNEL_GRID,
        samples: int = STD_SAMPLE_POINTS,
):
    r = np.linspace(0.0, cutoff, grid, dtype=np.float64)

    # 1. Плотность взаимодействия (константа внутри сферы, 0 снаружи)
    volume = (4.0 / 3.0) * math.pi * (sigma ** 3)
    density_value = 1.0 / volume

    w = np.zeros_like(r)
    w[r <= sigma] = density_value

    # 2. Вычисление радиусов для спавна потомков
    # Берем квантили от 0 до 1
    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples

    # Для равномерного распределения в 3D извлекаем кубический корень
    radii = sigma * np.cbrt(quantiles)
    radii = np.clip(radii, 0.0, cutoff)

    # Форматирование под архитектуру симулятора
    # Форматирование под архитектуру ISMD (плоские 1D массивы)
    return quantiles, radii, r, w

def _sample_mean_population(sim, warmup_events: int, block_events: int, blocks: int):
    if warmup_events > 0:
        sim.run_events(warmup_events)
    samples = np.empty(blocks, dtype=np.float64)
    for i in range(blocks):
        sim.run_events(block_events)
        samples[i] = sim.current_population()
    return float(np.mean(samples)), samples


def test_equal_kernels_1d_mean_field():
    b = STD_B
    d = STD_D
    dd = STD_DD
    L = 1000.0
    cells = 720

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_1d()
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[STD_CUTOFF]]),
        seed=123,
        cell_capacity=2048,
        is_periodic=True,
    )

    exp_N = int(((b - d) / dd) * L)
    assert exp_N == STD_EQ_POPULATION

    rng = np.random.default_rng(123)
    xs = rng.uniform(0.0, L, size=exp_N)
    for x in xs:
        sim.spawn_particle(0, float(x))

    mean_N, samples = _sample_mean_population(sim, warmup_events=7_000, block_events=3_500, blocks=2)
    assert abs(mean_N - exp_N) <= EQ_TOLERANCE * exp_N, (
        f"1D equal-kernel: mean N={mean_N:.1f}, expected {exp_N}, samples={samples}"
    )


def test_equal_kernels_2d_mean_field():
    b = STD_B
    d = STD_D
    dd = STD_DD
    cutoff = STD_CUTOFF
    Lx, Ly = 25.0, 40.0
    Nx, Ny = 48, 80

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_2d(cutoff=cutoff)
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
        seed=321,
        cell_capacity=256,
        is_periodic=True,
    )

    volume = Lx * Ly
    exp_N = int(((b - d) / dd) * volume)
    assert exp_N == STD_EQ_POPULATION

    rng = np.random.default_rng(321)
    xs = rng.uniform(0.0, Lx, size=exp_N)
    ys = rng.uniform(0.0, Ly, size=exp_N)
    for x, y in zip(xs, ys):
        pos = np.array([float(x), float(y)], dtype=np.float64)
        sim.spawn_particle(0, float(pos[0]) if pos.shape[0] == 1 else float(pos[0]), *(pos[1:].tolist()))

    mean_N, samples = _sample_mean_population(sim, warmup_events=8_000, block_events=4_000, blocks=2)
    assert abs(mean_N - exp_N) <= EQ_TOLERANCE * exp_N, (
        f"2D equal-kernel: mean N={mean_N:.1f}, expected {exp_N}, samples={samples}"
    )


def test_equal_kernels_3d_mean_field():
    b = STD_B
    d = STD_D
    dd = STD_DD
    cutoff = STD_CUTOFF
    Lx = Ly = Lz = 10.0
    Nx = Ny = Nz = 14

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_3d(cutoff=cutoff)
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
        seed=654,
        cell_capacity=192,
        is_periodic=True,
    )

    volume = Lx * Ly * Lz
    exp_N = int(((b - d) / dd) * volume)
    assert exp_N == STD_EQ_POPULATION

    rng = np.random.default_rng(654)
    xs = rng.uniform(0.0, Lx, size=exp_N)
    ys = rng.uniform(0.0, Ly, size=exp_N)
    zs = rng.uniform(0.0, Lz, size=exp_N)
    for x, y, z in zip(xs, ys, zs):
        pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
        sim.spawn_particle(0, float(pos[0]) if pos.shape[0] == 1 else float(pos[0]), *(pos[1:].tolist()))

    mean_N, samples = _sample_mean_population(sim, warmup_events=8_500, block_events=4_200, blocks=2)
    assert abs(mean_N - exp_N) <= EQ_TOLERANCE * exp_N, (
        f"3D equal-kernel: mean N={mean_N:.1f}, expected {exp_N}, samples={samples}"
    )


__all__ = [
    "half_normal_equal_tables_1d",
    "half_normal_equal_tables_2d",
    "half_normal_equal_tables_3d",
    "STD_B",
    "STD_D",
    "STD_DD",
    "STD_CUTOFF",
    "STD_EQ_POPULATION",
    "EQ_TOLERANCE",
    "STD_SIGMA",
    "STD_KERNEL_GRID",
    "STD_SAMPLE_POINTS",
]