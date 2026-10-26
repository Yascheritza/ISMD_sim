

"""
ISMD simulator based on Omelyan and Kozitsky's numerical method
"""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from numba import njit, types
from numba.experimental import jitclass
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp

# ============================================================================
# Helper Functions (JIT compiled for performance)
# ============================================================================

@njit(cache=True, inline='always')
def _interp_uniform(
        xdat: NDArray[np.float64],
        ydat: NDArray[np.float64],
        x: float,
        inv_dx: float,
) -> float:
    """Linear interpolation on uniformly spaced grid."""
    length = xdat.shape[0]
    if length == 1:
        return ydat[0]
    rel = x * inv_dx
    if rel <= 0.0:
        return ydat[0]
    upper = length - 1
    if rel >= upper:
        return ydat[upper]
    idx = int(rel)
    frac = rel - idx
    return ydat[idx] + (ydat[idx + 1] - ydat[idx]) * frac


@njit(cache=True, inline='always')
def _distance_1d(pos_a: float, pos_b: float, area_len: float, periodic: bool) -> float:
    """Calculate 1D distance with optional periodic boundaries."""
    diff = abs(pos_a - pos_b)
    if periodic and area_len > 0.0:
        wrap = area_len - diff
        if wrap < diff:
            diff = wrap
    return diff


@njit(cache=True, inline='always')
def _distance_2d(
        x1: float, y1: float, x2: float, y2: float,
        area_x: float, area_y: float, periodic: bool
) -> float:
    """Calculate 2D Euclidean distance with optional periodic boundaries."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    if periodic:
        if area_x > 0.0:
            wrap_x = area_x - dx
            if wrap_x < dx:
                dx = wrap_x
        if area_y > 0.0:
            wrap_y = area_y - dy
            if wrap_y < dy:
                dy = wrap_y

    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True, inline='always')
def _distance_3d(
        x1: float, y1: float, z1: float,
        x2: float, y2: float, z2: float,
        area_x: float, area_y: float, area_z: float,
        periodic: bool
) -> float:
    """Calculate 3D Euclidean distance with optional periodic boundaries."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)

    if periodic:
        if area_x > 0.0:
            wrap_x = area_x - dx
            if wrap_x < dx:
                dx = wrap_x
        if area_y > 0.0:
            wrap_y = area_y - dy
            if wrap_y < dy:
                dy = wrap_y
        if area_z > 0.0:
            wrap_z = area_z - dz
            if wrap_z < dz:
                dz = wrap_z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True)
def _build_kernel_matrix_nd(
        ndim: int,
        cell_counts: NDArray[np.int32],
        area_size: NDArray[np.float64],
        kernel_x: NDArray[np.float64],
        kernel_y: NDArray[np.float64],
        inv_dx: float,
        periodic: bool
) -> NDArray[np.float64]:
    """
    Creates matrix N_total x N_total for ISMD.
    Supports 1D, 2D и 3D.
    """
    total_cells = 1
    for d in range(ndim):
        total_cells *= cell_counts[d]

    matrix = np.zeros((total_cells, total_cells), dtype=np.float64)
    cell_size = np.empty(ndim, dtype=np.float64)
    for d in range(ndim):
        cell_size[d] = area_size[d] / float(cell_counts[d])

    for i in range(total_cells):
        pos_i = np.empty(ndim, dtype=np.float64)
        if ndim == 1:
            pos_i[0] = (i + 0.5) * cell_size[0]
        elif ndim == 2:
            ix = i % cell_counts[0]
            iy = i // cell_counts[0]
            pos_i[0] = (ix + 0.5) * cell_size[0]
            pos_i[1] = (iy + 0.5) * cell_size[1]
        else:  # 3D
            ix = i % cell_counts[0]
            temp = i // cell_counts[0]
            iy = temp % cell_counts[1]
            iz = temp // cell_counts[1]
            pos_i[0] = (ix + 0.5) * cell_size[0]
            pos_i[1] = (iy + 0.5) * cell_size[1]
            pos_i[2] = (iz + 0.5) * cell_size[2]

        for j in range(total_cells):
            pos_j = np.empty(ndim, dtype=np.float64)
            if ndim == 1:
                pos_j[0] = (j + 0.5) * cell_size[0]
            elif ndim == 2:
                jx = j % cell_counts[0]
                jy = j // cell_counts[0]
                pos_j[0] = (jx + 0.5) * cell_size[0]
                pos_j[1] = (jy + 0.5) * cell_size[1]
            else:  # 3D
                jx = j % cell_counts[0]
                temp = j // cell_counts[0]
                jy = temp % cell_counts[1]
                jz = temp // cell_counts[1]
                pos_j[0] = (jx + 0.5) * cell_size[0]
                pos_j[1] = (jy + 0.5) * cell_size[1]
                pos_j[2] = (jz + 0.5) * cell_size[2]

            dist = 0.0
            if ndim == 1:
                dist = _distance_1d(pos_i[0], pos_j[0], area_size[0], periodic)
            elif ndim == 2:
                dist = _distance_2d(pos_i[0], pos_i[1], pos_j[0], pos_j[1], area_size[0], area_size[1], periodic)
            else:
                dist = _distance_3d(pos_i[0], pos_i[1], pos_i[2], pos_j[0], pos_j[1], pos_j[2], area_size[0],
                                    area_size[1], area_size[2], periodic)

            val = _interp_uniform(kernel_x, kernel_y, dist, inv_dx)
            matrix[i, j] = val

    return matrix
# ============================================================================
# Main ISMD State Class, GPU version
# ============================================================================

class ISMDState:
    """
    ISMD method using KSA and decomposition propagation
    Runs on GPU using CuPy.
    """

    def __init__(
            self,
            ndim: np.int32,
            area_size: NDArray[np.float64],
            cell_counts: NDArray[np.int32],
            periodic: bool,
            m: float,
            birth_x: NDArray[np.float64],
            birth_y: NDArray[np.float64],
            death_x: NDArray[np.float64],
            death_y: NDArray[np.float64],
            n_init: NDArray[np.float64],
            u_init: NDArray[np.float64]
    ):
        self.ndim = ndim
        self.area_size = area_size
        self.cell_counts = cell_counts
        self.m = m
        self.time = 0.0

        self.total_cells = 1
        for d in range(ndim):
            self.total_cells *= cell_counts[d]

        self.h_volume = 1.0
        for d in range(ndim):
            self.h_volume *= (area_size[d] / float(cell_counts[d]))

        b_inv_dx = 1.0 / (birth_x[1] - birth_x[0]) if birth_x.shape[0] > 1 else 0.0
        d_inv_dx = 1.0 / (death_x[1] - death_x[0]) if death_x.shape[0] > 1 else 0.0

        a_mat_cpu = _build_kernel_matrix_nd(
            ndim, cell_counts, area_size, birth_x, birth_y, b_inv_dx, periodic
        )
        b_mat_cpu = _build_kernel_matrix_nd(
            ndim, cell_counts, area_size, death_x, death_y, d_inv_dx, periodic
        )

        self.a_mat = cp.asarray(a_mat_cpu, dtype=cp.float64)
        self.b_mat = cp.asarray(b_mat_cpu, dtype=cp.float64)
        self.n = cp.asarray(n_init, dtype=cp.float64)
        self.u = cp.asarray(u_init, dtype=cp.float64)

    @staticmethod
    def _linear_flow(value, alpha, beta, duration: float, tiny: float):
        small_beta = cp.abs(beta) < tiny
        safe_beta = cp.where(small_beta, 1.0, beta)
        decay = cp.exp(-beta * duration)
        regular = value * decay - cp.expm1(-beta * duration) * alpha / safe_beta
        return cp.where(small_beta, value + alpha * duration, regular)

    def _advance_density(self, index: int, duration: float, tiny: float) -> None:
        h = self.h_volume
        alpha = h * (
            cp.dot(self.a_mat[index], self.n) - self.a_mat[index, index] * self.n[index]
            - cp.dot(self.b_mat[index], self.u[index])
        )
        beta = self.m - h * self.a_mat[index, index]
        self.n[index] = self._linear_flow(self.n[index], alpha, beta, duration, tiny)

    def _pair_coefficients(self, i: int, j: int, tiny: float):
        h = self.h_volume
        n_safe = cp.where(self.n > 0.0, self.n, tiny)
        ni = n_safe[i]
        nj = n_safe[j]

        alpha = self.a_mat[i, j] * (self.n[i] + self.n[j])
        alpha += h * (
            cp.dot(self.a_mat[i], self.u[j]) - self.a_mat[i, i] * self.u[j, i]
        )
        alpha += h * (
            cp.dot(self.a_mat[j], self.u[i]) - self.a_mat[j, j] * self.u[i, j]
        )

        beta_i = self.m - h * self.a_mat[i, i]
        ksa_sum = (
            (self.b_mat[i] + self.b_mat[j])
            * self.u[i]
            * self.u[j]
            / n_safe
        )
        ksa_sum[i] = 0.0
        ksa_sum[j] = 0.0
        beta = 2.0 * (beta_i + self.b_mat[i, j]) + h * cp.sum(ksa_sum) / (ni * nj)

        gamma = h * (self.b_mat[i, i] + self.b_mat[i, j]) / (ni * nj)
        gamma *= self.u[i, i] / ni + self.u[j, j] / nj
        zeta = 2.0 * h * self.b_mat[i, i] / (ni ** 3)
        return alpha, beta, gamma, zeta

    def _advance_pair(self, i: int, j: int, duration: float, tiny: float) -> None:
        alpha, beta, gamma, zeta = self._pair_coefficients(i, j, tiny)
        value = self.u[i, j]

        # Симметричное разложение (19): нелинейная половина, линейный шаг, ещё половина.
        if i == j:
            value = value / cp.sqrt(1.0 + zeta * value * value * duration)
        else:
            value = value / (1.0 + gamma * value * duration / 2.0)

        value = self._linear_flow(value, alpha, beta, duration, tiny)

        if i == j:
            value = value / cp.sqrt(1.0 + zeta * value * value * duration)
        else:
            value = value / (1.0 + gamma * value * duration / 2.0)

        self.u[i, j] = value
        self.u[j, i] = value

    def step(self, dt: float) -> None:
        """Выполняет симметричный DP-шаг второго порядка из уравнения (17)."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        half_dt = dt / 2.0
        tiny = 1e-15

        for i in range(self.total_cells):
            for j in range(i, self.total_cells):
                self._advance_pair(i, j, half_dt, tiny)

        for i in range(self.total_cells - 1, -1, -1):
            self._advance_density(i, half_dt, tiny)
        for i in range(self.total_cells):
            self._advance_density(i, half_dt, tiny)

        for i in range(self.total_cells - 1, -1, -1):
            for j in range(self.total_cells - 1, i - 1, -1):
                self._advance_pair(i, j, half_dt, tiny)

        self.time += dt

    def get_density(self) -> NDArray[np.float64]:
        """
        Returns current density array n(x) for graphics; returns data on CPU after GPU
        """
        return self.n.get()


# ============================================================================
# Factory Functions for ISMD
# ============================================================================

def make_ismd_state(
        ndim: int,
        area_size: Sequence[float] | NDArray[np.float64],
        cell_counts: Sequence[int] | NDArray[np.int32],
        m: float,
        birth_x: NDArray[np.float64],
        birth_y: NDArray[np.float64],
        death_x: NDArray[np.float64],
        death_y: NDArray[np.float64],
        periodic: bool = True,
        n_init: NDArray[np.float64] | None = None,
        u_init: NDArray[np.float64] | None = None
) -> ISMDState:
    """
    Makes ISMD state (1D, 2D, 3D are supported).
    """
    area_arr = np.ascontiguousarray(area_size, dtype=np.float64)
    cells_arr = np.ascontiguousarray(cell_counts, dtype=np.int32)

    b_x = np.ascontiguousarray(birth_x, dtype=np.float64)
    b_y = np.ascontiguousarray(birth_y, dtype=np.float64)
    d_x = np.ascontiguousarray(death_x, dtype=np.float64)
    d_y = np.ascontiguousarray(death_y, dtype=np.float64)

    total_cells = int(np.prod(cells_arr))

    if n_init is None:
        n_array = np.zeros(total_cells, dtype=np.float64)
    else:
        n_array = np.ascontiguousarray(n_init, dtype=np.float64)

    if u_init is None:
        u_array = np.zeros((total_cells, total_cells), dtype=np.float64)
    else:
        u_array = np.ascontiguousarray(u_init, dtype=np.float64)

    return ISMDState(
        ndim=np.int32(ndim),
        area_size=area_arr,
        cell_counts=cells_arr,
        periodic=periodic,
        m=float(m),
        birth_x=b_x,
        birth_y=b_y,
        death_x=d_x,
        death_y=d_y,
        n_init=n_array,
        u_init=u_array
    )


__all__ = [
    "ISMDState",
    "make_ismd_state",
]
