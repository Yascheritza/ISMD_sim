from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray


def radial_profile_3d(
    density: NDArray[np.float64],
    area_size: Sequence[float],
    cell_counts: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Усредняет трёхмерную плотность по сферическим слоям внутри вписанной сферы."""
    area = np.asarray(area_size, dtype=np.float64)
    cells = np.asarray(cell_counts, dtype=np.int64)
    if area.shape != (3,) or cells.shape != (3,):
        raise ValueError("area_size and cell_counts must contain three values")
    if np.any(area <= 0.0) or np.any(cells <= 0):
        raise ValueError("area sizes and cell counts must be positive")

    nx, ny, nz = (int(value) for value in cells)
    values = np.asarray(density, dtype=np.float64)
    if values.size != nx * ny * nz:
        raise ValueError("density size does not match cell_counts")

    dx, dy, dz = area / cells
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx - area[0] / 2.0
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy - area[1] / 2.0
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz - area[2] / 2.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy + zz * zz).ravel()

    shell_width = float(min(dx, dy, dz))
    max_radius = float(area.min() / 2.0)
    shell_count = max(1, int(math.ceil(max_radius / shell_width)))
    inside = radius < max_radius
    shell_index = np.minimum((radius[inside] / shell_width).astype(np.int64), shell_count - 1)
    shell_sum = np.bincount(shell_index, weights=values.ravel()[inside], minlength=shell_count)
    shell_size = np.bincount(shell_index, minlength=shell_count)

    valid = shell_size > 0
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    profile = np.empty(shell_count, dtype=np.float64)
    profile[valid] = shell_sum[valid] / shell_size[valid]
    missing = ~valid
    if np.any(missing):
        indices = np.arange(shell_count, dtype=np.float64)
        profile[missing] = np.interp(indices[missing], indices[valid], profile[valid])

    radii = (np.arange(shell_count, dtype=np.float64) + 0.5) * shell_width
    return radii, profile


def front_radii_at_levels(
    radii: NDArray[np.float64],
    density: NDArray[np.float64],
    levels: Sequence[float],
) -> NDArray[np.float64]:
    """Находит внешние пересечения профиля с долями его текущего максимума."""
    radii = np.asarray(radii, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    fractions = np.asarray(levels, dtype=np.float64)
    if radii.ndim != 1 or density.shape != radii.shape:
        raise ValueError("radii and density must be one-dimensional arrays of equal size")
    if np.any((fractions <= 0.0) | (fractions >= 1.0)):
        raise ValueError("front levels must be between zero and one")

    result = np.full(fractions.shape, np.nan, dtype=np.float64)
    if radii.size < 2 or not np.any(np.isfinite(density)):
        return result
    maximum = float(np.nanmax(density))
    if maximum <= 0.0:
        return result

    for level_index, fraction in enumerate(fractions):
        level = fraction * maximum
        crossings = np.flatnonzero((density[:-1] >= level) & (density[1:] < level))
        if crossings.size == 0:
            continue
        index = int(crossings[-1])
        left_value = density[index]
        right_value = density[index + 1]
        if right_value == left_value:
            result[level_index] = radii[index]
        else:
            weight = (level - left_value) / (right_value - left_value)
            result[level_index] = radii[index] + weight * (radii[index + 1] - radii[index])
    return result


def article_front_speeds(
    radii: NDArray[np.float64],
    previous_density: NDArray[np.float64],
    density: NDArray[np.float64],
    delta_time: float,
    front_radii: NDArray[np.float64],
    gradient_floor: float = 1e-12,
) -> NDArray[np.float64]:
    """Вычисляет v=-∂t n/∂r n в заданных точках фронта."""
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    radii = np.asarray(radii, dtype=np.float64)
    previous = np.asarray(previous_density, dtype=np.float64)
    current = np.asarray(density, dtype=np.float64)
    positions = np.asarray(front_radii, dtype=np.float64)
    if radii.ndim != 1 or previous.shape != radii.shape or current.shape != radii.shape:
        raise ValueError("radial profiles must have equal one-dimensional shapes")
    if radii.size < 2:
        return np.full(positions.shape, np.nan, dtype=np.float64)

    time_derivative = (current - previous) / delta_time
    radial_derivative = np.gradient(current, radii)
    result = np.full(positions.shape, np.nan, dtype=np.float64)
    for index, position in np.ndenumerate(positions):
        if not np.isfinite(position) or position < radii[0] or position > radii[-1]:
            continue
        dn_dt = float(np.interp(position, radii, time_derivative))
        dn_dr = float(np.interp(position, radii, radial_derivative))
        if abs(dn_dr) >= gradient_floor:
            result[index] = -dn_dt / dn_dr
    return result


def linear_spreading_speed_3d(
    kernel: str,
    c_plus: float,
    mortality: float,
    sigma: float,
) -> tuple[float, float]:
    """Даёт линейную MF-оценку скорости и λ для плоского края 3D-фронта."""
    from scipy.optimize import minimize_scalar

    if c_plus <= mortality:
        raise ValueError("c_plus must exceed mortality for an invading front")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    def moment_generating_function(value: float) -> float:
        if kernel == "gaussian":
            return math.exp(0.5 * value * value)
        if kernel == "tophat":
            if abs(value) < 1e-4:
                return 1.0 + value * value / 10.0 + value ** 4 / 280.0
            return 3.0 * (value * math.cosh(value) - math.sinh(value)) / (value ** 3)
        raise ValueError("kernel must be 'gaussian' or 'tophat'")

    result = minimize_scalar(
        lambda scaled_lambda: sigma
        * (c_plus * moment_generating_function(scaled_lambda) - mortality)
        / scaled_lambda,
        bounds=(1e-6, 10.0),
        method="bounded",
    )
    return float(result.fun), float(result.x / sigma)


__all__ = [
    "article_front_speeds",
    "front_radii_at_levels",
    "linear_spreading_speed_3d",
    "radial_profile_3d",
]
