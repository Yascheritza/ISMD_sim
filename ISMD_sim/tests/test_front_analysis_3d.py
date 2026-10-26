import numpy as np

from SSA.front_analysis import (
    article_front_speeds,
    front_radii_at_levels,
    linear_spreading_speed_3d,
    radial_profile_3d,
)


def test_radial_profile_preserves_radial_order():
    area = np.array([8.0, 8.0, 8.0])
    cells = np.array([16, 16, 16])
    coordinates = (np.arange(16, dtype=np.float64) + 0.5) * 0.5 - 4.0
    zz, yy, xx = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = np.exp(-(xx * xx + yy * yy + zz * zz) / 2.0)

    radii, profile = radial_profile_3d(density.ravel(), area, cells)

    assert radii.shape == profile.shape
    assert profile[0] > profile[-1]
    assert np.count_nonzero(np.diff(profile) > 0.02) == 0


def test_front_radius_uses_outermost_crossing():
    radii = np.arange(6, dtype=np.float64)
    density = np.array([1.0, 0.4, 0.8, 0.6, 0.2, 0.0])

    front = front_radii_at_levels(radii, density, [0.5])

    assert np.isclose(front[0], 3.25)


def test_article_speed_recovers_travelling_wave_velocity():
    radii = np.linspace(0.0, 20.0, 1001)
    velocity = 1.7
    delta_time = 0.01
    previous = 1.0 / (1.0 + np.exp((radii - 8.0) / 0.7))
    current = 1.0 / (1.0 + np.exp((radii - (8.0 + velocity * delta_time)) / 0.7))
    front = front_radii_at_levels(radii, current, [0.25, 0.5, 0.75])

    speed = article_front_speeds(radii, previous, current, delta_time, front)

    assert np.allclose(speed, velocity, atol=0.04)


def test_linear_speed_has_expected_parameter_dependence():
    base_speed, _ = linear_spreading_speed_3d("gaussian", c_plus=1.0, mortality=0.01, sigma=1.0)
    narrow_speed, _ = linear_spreading_speed_3d("gaussian", c_plus=1.0, mortality=0.01, sigma=0.5)
    high_birth_speed, _ = linear_spreading_speed_3d("gaussian", c_plus=1.2, mortality=0.01, sigma=1.0)
    high_mortality_speed, _ = linear_spreading_speed_3d("gaussian", c_plus=1.0, mortality=0.1, sigma=1.0)

    assert np.isclose(narrow_speed, 0.5 * base_speed)
    assert high_birth_speed > base_speed
    assert high_mortality_speed < base_speed


def test_article_parameter_linear_speed_estimates():
    type1_speed, type1_lambda = linear_spreading_speed_3d("tophat", 1.0, 0.01, 0.1)
    type2_speed, type2_lambda = linear_spreading_speed_3d("gaussian", 1.0, 0.01, 1.0)

    assert np.isclose(type1_speed, 0.070749, rtol=1e-5)
    assert np.isclose(type1_lambda, 23.91325, rtol=1e-5)
    assert np.isclose(type2_speed, 1.638706, rtol=1e-5)
    assert np.isclose(type2_lambda, 0.996953, rtol=1e-5)
