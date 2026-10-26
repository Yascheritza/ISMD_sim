import math

import numpy as np

from SSA.model_kernels import gaussian_kernel_3d, tophat_kernel_3d


def _radial_integral(radius, values):
    return 4.0 * math.pi * np.trapezoid(radius * radius * values, radius)


def test_gaussian_kernel_has_article_normalization():
    radius, values = gaussian_kernel_3d(sigma=0.7, intensity=1.3, grid_size=4096)

    assert np.isclose(_radial_integral(radius, values), 1.3, rtol=3e-5)


def test_tophat_kernel_has_article_normalization():
    radius, values = tophat_kernel_3d(sigma=0.7, intensity=1.3, grid_size=4096)

    assert np.isclose(_radial_integral(radius, values), 1.3, rtol=5e-4)
