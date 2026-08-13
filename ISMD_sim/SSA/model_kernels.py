from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def gaussian_kernel_3d(
    sigma: float,
    intensity: float = 1.0,
    grid_size: int = 512,
    cutoff_sigmas: float = 8.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Таблица 3D-ядра c/(2πσ²)^(3/2) exp(-r²/(2σ²))."""
    if sigma <= 0.0 or intensity < 0.0 or grid_size < 2 or cutoff_sigmas <= 0.0:
        raise ValueError("invalid Gaussian kernel parameters")
    cutoff = cutoff_sigmas * sigma
    radius = np.linspace(0.0, cutoff, grid_size, dtype=np.float64)
    normalization = intensity / ((2.0 * math.pi * sigma * sigma) ** 1.5)
    values = normalization * np.exp(-(radius * radius) / (2.0 * sigma * sigma))
    step = radius[1] - radius[0]
    return np.append(radius, radius[-1] + step), np.append(values, 0.0)


def tophat_kernel_3d(
    sigma: float,
    intensity: float = 1.0,
    grid_size: int = 512,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Равномерное ядро в 3D-шаре с нормировкой ∫a(x,y)dy=intensity."""
    if sigma <= 0.0 or intensity < 0.0 or grid_size < 2:
        raise ValueError("invalid top-hat kernel parameters")
    radius = np.linspace(0.0, sigma, grid_size, dtype=np.float64)
    values = np.full(radius.shape, intensity / ((4.0 / 3.0) * math.pi * sigma ** 3))
    step = radius[1] - radius[0]
    return np.append(radius, radius[-1] + step), np.append(values, 0.0)


__all__ = ["gaussian_kernel_3d", "tophat_kernel_3d"]
