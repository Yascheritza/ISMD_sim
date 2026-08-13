import importlib
import math
import sys
import types

import numpy as np


def _load_ismd_with_numpy_backend(monkeypatch):
    cupyx = types.ModuleType("cupyx")
    cupyx.__path__ = []
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_scipy.__path__ = []
    cupyx_sparse = types.ModuleType("cupyx.scipy.sparse")

    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", cupyx_sparse)
    sys.modules.pop("SSA.ismd_sim", None)
    return importlib.import_module("SSA.ismd_sim")


def test_single_cell_density_flow_uses_equation_14_beta(monkeypatch):
    ismd_sim = _load_ismd_with_numpy_backend(monkeypatch)
    state = ismd_sim.make_ismd_state(
        ndim=1,
        area_size=[1.0],
        cell_counts=[1],
        m=0.5,
        birth_x=np.array([0.0, 1.0]),
        birth_y=np.array([0.2, 0.0]),
        death_x=np.array([0.0, 1.0]),
        death_y=np.array([0.0, 0.0]),
        periodic=False,
        n_init=np.array([1.0]),
        u_init=np.zeros((1, 1)),
    )

    state.step(0.4)

    expected = math.exp(-(0.5 - 0.2) * 0.4)
    assert np.isclose(state.n[0], expected)
    assert np.isclose(state.time, 0.4)


def test_pair_coefficients_match_equation_15(monkeypatch):
    ismd_sim = _load_ismd_with_numpy_backend(monkeypatch)
    state = ismd_sim.make_ismd_state(
        ndim=1,
        area_size=[2.0],
        cell_counts=[2],
        m=0.1,
        birth_x=np.array([0.0, 1.0, 2.0]),
        birth_y=np.array([0.4, 0.2, 0.0]),
        death_x=np.array([0.0, 1.0, 2.0]),
        death_y=np.array([0.3, 0.1, 0.0]),
        periodic=False,
        n_init=np.array([1.2, 0.8]),
        u_init=np.array([[1.1, 0.7], [0.7, 0.6]]),
    )

    alpha, beta, gamma, _ = state._pair_coefficients(0, 1, 1e-15)
    h = state.h_volume
    a = state.a_mat
    b = state.b_mat
    n = state.n
    u = state.u
    expected_alpha = a[0, 1] * (n[0] + n[1])
    expected_alpha += h * a[0, 1] * u[1, 1]
    expected_alpha += h * a[1, 0] * u[0, 0]
    expected_beta = 2.0 * (state.m - h * a[0, 0] + b[0, 1])
    expected_gamma = h * (b[0, 0] + b[0, 1]) / (n[0] * n[1])
    expected_gamma *= u[0, 0] / n[0] + u[1, 1] / n[1]

    assert np.isclose(alpha, expected_alpha)
    assert np.isclose(beta, expected_beta)
    assert np.isclose(gamma, expected_gamma)


def test_article_step_preserves_pair_symmetry(monkeypatch):
    ismd_sim = _load_ismd_with_numpy_backend(monkeypatch)
    n_init = np.array([1.0, 0.9])
    state = ismd_sim.make_ismd_state(
        ndim=1,
        area_size=[2.0],
        cell_counts=[2],
        m=0.01,
        birth_x=np.array([0.0, 1.0, 2.0]),
        birth_y=np.array([0.4, 0.2, 0.0]),
        death_x=np.array([0.0, 1.0, 2.0]),
        death_y=np.array([0.2, 0.1, 0.0]),
        periodic=False,
        n_init=n_init,
        u_init=np.outer(n_init, n_init),
    )

    state.step(0.01)

    assert np.all(np.isfinite(state.n))
    assert np.all(np.isfinite(state.u))
    assert np.all(state.n > 0.0)
    assert np.all(state.u > 0.0)
    assert np.allclose(state.u, state.u.T)


def test_3d_step_has_the_article_equations_as_small_dt_limit(monkeypatch):
    ismd_sim = _load_ismd_with_numpy_backend(monkeypatch)
    n_init = np.array([1.1, 0.9])
    u_init = np.array([[1.0, 0.75], [0.75, 0.7]])
    state = ismd_sim.make_ismd_state(
        ndim=3,
        area_size=[2.0, 1.0, 1.0],
        cell_counts=[2, 1, 1],
        m=0.03,
        birth_x=np.array([0.0, 1.0, 2.0]),
        birth_y=np.array([0.5, 0.25, 0.0]),
        death_x=np.array([0.0, 1.0, 2.0]),
        death_y=np.array([0.2, 0.08, 0.0]),
        periodic=False,
        n_init=n_init.copy(),
        u_init=u_init.copy(),
    )
    h = state.h_volume
    a = state.a_mat.copy()
    b = state.b_mat.copy()
    expected_n = -state.m * n_init + h * (a @ n_init - np.sum(b * u_init, axis=1))
    expected_u = np.empty_like(u_init)
    for i in range(2):
        for j in range(2):
            value = -2.0 * (state.m + b[i, j]) * u_init[i, j]
            value += a[i, j] * (n_init[i] + n_init[j])
            for k in range(2):
                w_ijk = u_init[i, j] * u_init[i, k] * u_init[j, k]
                w_ijk /= n_init[i] * n_init[j] * n_init[k]
                value += h * (
                    a[i, k] * u_init[j, k]
                    + a[j, k] * u_init[i, k]
                    - (b[i, k] + b[j, k]) * w_ijk
                )
            expected_u[i, j] = value

    dt = 1e-7
    state.step(dt)

    assert np.allclose((state.n - n_init) / dt, expected_n, rtol=2e-6, atol=2e-7)
    assert np.allclose((state.u - u_init) / dt, expected_u, rtol=2e-6, atol=2e-7)


def test_competition_intensity_and_range_affect_3d_dynamics(monkeypatch):
    ismd_sim = _load_ismd_with_numpy_backend(monkeypatch)

    def make_state(death_y):
        n_init = np.array([1.0, 0.8])
        return ismd_sim.make_ismd_state(
            ndim=3,
            area_size=[2.0, 1.0, 1.0],
            cell_counts=[2, 1, 1],
            m=0.01,
            birth_x=np.array([0.0, 1.0, 2.0]),
            birth_y=np.array([0.5, 0.2, 0.0]),
            death_x=np.array([0.0, 1.0, 2.0]),
            death_y=np.asarray(death_y),
            periodic=False,
            n_init=n_init,
            u_init=np.outer(n_init, n_init),
        )

    base = make_state([0.2, 0.05, 0.0])
    stronger = make_state([0.4, 0.1, 0.0])
    longer_range = make_state([0.2, 0.15, 0.0])
    for state in (base, stronger, longer_range):
        state.step(0.01)

    assert np.all(stronger.n < base.n)
    assert not np.allclose(longer_range.n, base.n)
