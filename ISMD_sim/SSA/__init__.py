"""
SSA - Spatial Stochastic Algorithm simulation package

This package provides high-performance spatial birth-death process simulations
using the Gillespie Stochastic Simulation Algorithm (SSA) based on the
Bolcker-Pacala-Dieckmann-Law model.

Features:
- 1D, 2D, and 3D spatial simulations
- Periodic and killing (absorbing) boundary conditions
- Event tracing for detailed analysis
- Automatic capacity management
- Optimized with Numba JIT compilation
"""

from .numba_sim import (
    SSAState,
    make_ssa_state_1d,
    make_ssa_state_2d,
    make_ssa_state_3d,
    get_all_particle_coords,
    get_all_particle_death_rates,
    EVENT_BIRTH,
    EVENT_DEATH,
    EVENT_MANUAL_SPAWN,
    EVENT_MANUAL_KILL,
)

__all__ = [
    "SSAState",
    "make_ssa_state_1d",
    "make_ssa_state_2d",
    "make_ssa_state_3d",
    "get_all_particle_coords",
    "get_all_particle_death_rates",
    "EVENT_BIRTH",
    "EVENT_DEATH",
    "EVENT_MANUAL_SPAWN",
    "EVENT_MANUAL_KILL",
]

__version__ = "2.0.0"
