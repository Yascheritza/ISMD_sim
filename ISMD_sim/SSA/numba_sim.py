"""
Spatial Stochastic Simulator using Gillespie SSA (Stochastic Simulation Algorithm)
Based on Bolcker-Pacala-Dieckmann-Law model

Features:
- Support for 1D, 2D, and 3D simulations
- Periodic and killing (absorbing) boundary conditions
- Event tracing for detailed simulation analysis
- Dynamic capacity management
- Optimized for maximum performance using Numba JIT compilation
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from numba import njit, types
from numba.experimental import jitclass

# Event type constants for tracing
EVENT_BIRTH = np.int32(0)
EVENT_DEATH = np.int32(1)
EVENT_MANUAL_SPAWN = np.int32(2)
EVENT_MANUAL_KILL = np.int32(3)


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


@njit(cache=True, inline='always')
def _sample_weighted(values: NDArray[np.float64]) -> int:
    """Sample an index from a weighted distribution."""
    n = values.shape[0]
    total = 0.0
    for idx in range(n):
        total += values[idx]
    if total <= 0.0:
        return -1
    r = np.random.random() * total
    cumulative = 0.0
    for idx in range(n):
        cumulative += values[idx]
        if r <= cumulative:
            return idx
    return n - 1


@njit(cache=True, inline='always')
def _wrap_position_1d(pos: float, area_len: float, periodic: bool) -> tuple:
    """Wrap position in 1D with boundary handling. Returns (wrapped_pos, is_valid)."""
    if periodic:
        if area_len > 0.0:
            wrapped = pos - area_len * math.floor(pos / area_len)
            if wrapped < 0.0:
                wrapped += area_len
            return (wrapped, True)
        return (pos, True)
    if pos < 0.0 or pos > area_len:
        return (pos, False)
    return (pos, True)


@njit(cache=True, inline='always')
def _wrap_position_2d(x: float, y: float, area_x: float, area_y: float, periodic: bool) -> tuple:
    """Wrap position in 2D with boundary handling. Returns (x, y, is_valid)."""
    if periodic:
        if area_x > 0.0:
            x = x - area_x * math.floor(x / area_x)
            if x < 0.0:
                x += area_x
        if area_y > 0.0:
            y = y - area_y * math.floor(y / area_y)
            if y < 0.0:
                y += area_y
        return (x, y, True)
    else:
        valid = (0.0 <= x <= area_x) and (0.0 <= y <= area_y)
        return (x, y, valid)


@njit(cache=True, inline='always')
def _wrap_position_3d(
    x: float, y: float, z: float,
    area_x: float, area_y: float, area_z: float,
    periodic: bool
) -> tuple:
    """Wrap position in 3D with boundary handling. Returns (x, y, z, is_valid)."""
    if periodic:
        if area_x > 0.0:
            x = x - area_x * math.floor(x / area_x)
            if x < 0.0:
                x += area_x
        if area_y > 0.0:
            y = y - area_y * math.floor(y / area_y)
            if y < 0.0:
                y += area_y
        if area_z > 0.0:
            z = z - area_z * math.floor(z / area_z)
            if z < 0.0:
                z += area_z
        return (x, y, z, True)
    else:
        valid = (0.0 <= x <= area_x) and (0.0 <= y <= area_y) and (0.0 <= z <= area_z)
        return (x, y, z, valid)


# ============================================================================
# Main SSA State Class
# ============================================================================

ssa_state_spec = [
    # Configuration
    ("ndim", types.int32),
    ("species_count", types.int32),
    ("area_size", types.Array(types.float64, 1, "C")),
    ("cell_counts", types.Array(types.int32, 1, "C")),
    ("cell_size", types.Array(types.float64, 1, "C")),
    ("periodic", types.boolean),
    ("seed", types.int64),
    
    # Rate parameters
    ("b", types.Array(types.float64, 1, "C")),
    ("d", types.Array(types.float64, 1, "C")),
    ("dd", types.Array(types.float64, 2, "C")),
    ("cutoff", types.Array(types.float64, 2, "C")),
    ("cull_range", types.Array(types.int32, 2, "C")),
    
    # Kernel interpolation tables
    ("birth_x", types.Array(types.float64, 2, "C")),
    ("birth_y", types.Array(types.float64, 2, "C")),
    ("birth_inv_dx", types.Array(types.float64, 1, "C")),
    ("death_x", types.Array(types.float64, 3, "C")),
    ("death_y", types.Array(types.float64, 3, "C")),
    ("death_inv_dx", types.Array(types.float64, 2, "C")),
    
    # Particle storage
    ("capacity", types.int64),
    ("population", types.int64),
    ("positions", types.Array(types.float64, 2, "C")),
    ("species_id", types.Array(types.int32, 1, "C")),
    ("death_rate", types.Array(types.float64, 1, "C")),
    
    # Cell-based spatial indexing
    ("total_cells", types.int64),
    ("cell_capacity", types.int32),
    ("particle_cell", types.Array(types.int64, 1, "C")),
    ("particle_slot", types.Array(types.int32, 1, "C")),
    ("cell_particles", types.Array(types.int64, 2, "C")),
    ("cell_particle_count", types.Array(types.int32, 1, "C")),
    
    # Aggregated rates per cell
    ("cell_species_count", types.Array(types.int32, 2, "C")),
    ("cell_birth_rate", types.Array(types.float64, 1, "C")),
    ("cell_death_rate", types.Array(types.float64, 1, "C")),
    ("cell_birth_rate_by_species", types.Array(types.float64, 2, "C")),
    ("cell_death_rate_by_species", types.Array(types.float64, 2, "C")),
    
    # Global rates
    ("total_birth_rate", types.float64),
    ("total_death_rate", types.float64),
    
    # Simulation state
    ("time", types.float64),
    ("event_count", types.int64),
    ("capacity_reached", types.boolean),
    
    # Trace storage (optional)
    ("trace_enabled", types.boolean),
    ("trace_capacity", types.int64),
    ("trace_count", types.int64),
    ("trace_type", types.Array(types.int32, 1, "C")),
    ("trace_species", types.Array(types.int32, 1, "C")),
    ("trace_time", types.Array(types.float64, 1, "C")),
    ("trace_pos", types.Array(types.float64, 2, "C")),
]


@jitclass(ssa_state_spec)
class SSAState:
    """
    Spatial Stochastic Simulation using Gillespie algorithm.
    
    Supports 1D, 2D, and 3D with periodic or killing boundaries.
    Uses cell-based spatial indexing for efficient neighbor searches.
    """
    
    def __init__(
        self,
        ndim: np.int32,
        species_count: np.int32,
        area_size: NDArray[np.float64],
        cell_counts: NDArray[np.int32],
        periodic: bool,
        capacity: np.int64,
        cell_capacity: np.int32,
        seed_value: np.int64,
        b: NDArray[np.float64],
        d: NDArray[np.float64],
        dd: NDArray[np.float64],
        cutoff: NDArray[np.float64],
        birth_x: NDArray[np.float64],
        birth_y: NDArray[np.float64],
        death_x: NDArray[np.float64],
        death_y: NDArray[np.float64],
        trace_enabled: bool,
        trace_capacity: np.int64,
    ):
        # Configuration
        self.ndim = ndim
        self.species_count = species_count
        self.area_size = area_size
        self.cell_counts = cell_counts
        self.periodic = periodic
        self.seed = seed_value
        
        # Rate parameters (assign first to avoid name collision)
        self.b = b
        self.d = d
        self.dd = dd
        self.cutoff = cutoff
        
        # Calculate cell sizes
        self.cell_size = np.empty(ndim, dtype=np.float64)
        for dim in range(ndim):
            self.cell_size[dim] = area_size[dim] / float(cell_counts[dim])
        
        # Calculate cull ranges
        self.cull_range = np.zeros((species_count, species_count), dtype=np.int32)
        for s1 in range(species_count):
            for s2 in range(species_count):
                if cutoff[s1, s2] > 0.0:
                    max_cell_size = 0.0
                    for dim in range(ndim):
                        if self.cell_size[dim] > max_cell_size:
                            max_cell_size = self.cell_size[dim]
                    self.cull_range[s1, s2] = int(math.ceil(cutoff[s1, s2] / max_cell_size))
        
        # Kernel tables
        self.birth_x = birth_x
        self.birth_y = birth_y
        self.death_x = death_x
        self.death_y = death_y
        
        # Precompute inverse dx for interpolation
        self.birth_inv_dx = np.empty(species_count, dtype=np.float64)
        birth_cols = birth_x.shape[1]
        for s in range(species_count):
            if birth_cols > 1:
                dx = birth_x[s, 1] - birth_x[s, 0]
                self.birth_inv_dx[s] = 1.0 / dx if dx != 0.0 else 0.0
            else:
                self.birth_inv_dx[s] = 0.0
        
        self.death_inv_dx = np.empty((species_count, species_count), dtype=np.float64)
        death_cols = death_x.shape[2]
        for s1 in range(species_count):
            for s2 in range(species_count):
                if death_cols > 1:
                    dx = death_x[s1, s2, 1] - death_x[s1, s2, 0]
                    self.death_inv_dx[s1, s2] = 1.0 / dx if dx != 0.0 else 0.0
                else:
                    self.death_inv_dx[s1, s2] = 0.0
        
        # Particle storage
        self.capacity = capacity
        self.population = np.int64(0)
        self.positions = np.zeros((capacity, ndim), dtype=np.float64)
        self.species_id = np.full(capacity, -1, dtype=np.int32)
        self.death_rate = np.zeros(capacity, dtype=np.float64)
        
        # Calculate total cells
        self.total_cells = np.int64(1)
        for d in range(ndim):
            self.total_cells *= cell_counts[d]
        
        # Cell indexing
        self.cell_capacity = cell_capacity
        self.particle_cell = np.full(capacity, -1, dtype=np.int64)
        self.particle_slot = np.full(capacity, -1, dtype=np.int32)
        self.cell_particles = np.full((self.total_cells, cell_capacity), -1, dtype=np.int64)
        self.cell_particle_count = np.zeros(self.total_cells, dtype=np.int32)
        
        # Cell rates
        self.cell_species_count = np.zeros((self.total_cells, species_count), dtype=np.int32)
        self.cell_birth_rate = np.zeros(self.total_cells, dtype=np.float64)
        self.cell_death_rate = np.zeros(self.total_cells, dtype=np.float64)
        self.cell_birth_rate_by_species = np.zeros((self.total_cells, species_count), dtype=np.float64)
        self.cell_death_rate_by_species = np.zeros((self.total_cells, species_count), dtype=np.float64)
        
        # Global state
        self.total_birth_rate = 0.0
        self.total_death_rate = 0.0
        self.time = 0.0
        self.event_count = np.int64(0)
        self.capacity_reached = False
        
        # Trace storage
        self.trace_enabled = trace_enabled
        self.trace_capacity = trace_capacity
        self.trace_count = np.int64(0)
        if trace_enabled:
            self.trace_type = np.empty(trace_capacity, dtype=np.int32)
            self.trace_species = np.empty(trace_capacity, dtype=np.int32)
            self.trace_time = np.empty(trace_capacity, dtype=np.float64)
            self.trace_pos = np.empty((trace_capacity, ndim), dtype=np.float64)
        else:
            self.trace_type = np.empty(0, dtype=np.int32)
            self.trace_species = np.empty(0, dtype=np.int32)
            self.trace_time = np.empty(0, dtype=np.float64)
            self.trace_pos = np.empty((0, ndim), dtype=np.float64)
        
        # Initialize random seed
        if seed_value >= 0:
            np.random.seed(int(seed_value))
    
    def _double_cell_capacity(self) -> None:
        """Double the cell capacity by reallocating the cell_particles array."""
        old_capacity = self.cell_capacity
        new_capacity = old_capacity * 2
        
        # Create new cell_particles array with doubled capacity
        new_cell_particles = np.full((self.total_cells, new_capacity), -1, dtype=np.int64)
        
        # Copy existing data
        for cell_id in range(self.total_cells):
            for slot in range(old_capacity):
                new_cell_particles[cell_id, slot] = self.cell_particles[cell_id, slot]
        
        # Update the array and capacity
        self.cell_particles = new_cell_particles
        self.cell_capacity = new_capacity
    
    def _double_particle_capacity(self) -> None:
        """Double the total particle capacity by reallocating particle storage arrays."""
        old_capacity = self.capacity
        new_capacity = old_capacity * 2
        
        # Create new arrays with doubled capacity
        new_positions = np.zeros((new_capacity, self.ndim), dtype=np.float64)
        new_species_id = np.full(new_capacity, -1, dtype=np.int32)
        new_death_rate = np.zeros(new_capacity, dtype=np.float64)
        new_particle_cell = np.full(new_capacity, -1, dtype=np.int64)
        new_particle_slot = np.full(new_capacity, -1, dtype=np.int32)
        
        # Copy existing data
        for idx in range(old_capacity):
            for d in range(self.ndim):
                new_positions[idx, d] = self.positions[idx, d]
            new_species_id[idx] = self.species_id[idx]
            new_death_rate[idx] = self.death_rate[idx]
            new_particle_cell[idx] = self.particle_cell[idx]
            new_particle_slot[idx] = self.particle_slot[idx]
        
        # Update arrays and capacity
        self.positions = new_positions
        self.species_id = new_species_id
        self.death_rate = new_death_rate
        self.particle_cell = new_particle_cell
        self.particle_slot = new_particle_slot
        self.capacity = new_capacity
    
    def _position_to_cell_id(self, pos: NDArray[np.float64]) -> int:
        """Convert position to linear cell index."""
        if self.ndim == 1:
            idx = int(pos[0] / self.cell_size[0])
            if idx < 0:
                idx = 0
            elif idx >= self.cell_counts[0]:
                idx = self.cell_counts[0] - 1
            return idx
        elif self.ndim == 2:
            ix = int(pos[0] / self.cell_size[0])
            iy = int(pos[1] / self.cell_size[1])
            if ix < 0:
                ix = 0
            elif ix >= self.cell_counts[0]:
                ix = self.cell_counts[0] - 1
            if iy < 0:
                iy = 0
            elif iy >= self.cell_counts[1]:
                iy = self.cell_counts[1] - 1
            return iy * self.cell_counts[0] + ix
        else:  # 3D
            ix = int(pos[0] / self.cell_size[0])
            iy = int(pos[1] / self.cell_size[1])
            iz = int(pos[2] / self.cell_size[2])
            if ix < 0:
                ix = 0
            elif ix >= self.cell_counts[0]:
                ix = self.cell_counts[0] - 1
            if iy < 0:
                iy = 0
            elif iy >= self.cell_counts[1]:
                iy = self.cell_counts[1] - 1
            if iz < 0:
                iz = 0
            elif iz >= self.cell_counts[2]:
                iz = self.cell_counts[2] - 1
            return (iz * self.cell_counts[1] + iy) * self.cell_counts[0] + ix
    
    def _get_neighbor_cells(self, cell_id: int, cull: int) -> tuple:
        """Get all neighboring cells within cull range. Returns (array, count)."""
        # Pre-allocate maximum possible neighbors
        max_neighbors = (2 * cull + 1) ** self.ndim
        neighbors = np.empty(max_neighbors, dtype=np.int64)
        count = 0
        
        if self.ndim == 1:
            cx = cell_id
            for offset in range(-cull, cull + 1):
                nx = cx + offset
                if self.periodic:
                    nx = nx % self.cell_counts[0]
                    neighbors[count] = np.int64(nx)
                    count += 1
                elif 0 <= nx < self.cell_counts[0]:
                    neighbors[count] = np.int64(nx)
                    count += 1
        
        elif self.ndim == 2:
            nx_cells = self.cell_counts[0]
            cx = cell_id % nx_cells
            cy = cell_id // nx_cells
            
            for dy in range(-cull, cull + 1):
                ny = cy + dy
                if self.periodic:
                    ny = ny % self.cell_counts[1]
                elif ny < 0 or ny >= self.cell_counts[1]:
                    continue
                
                for dx in range(-cull, cull + 1):
                    nx = cx + dx
                    if self.periodic:
                        nx = nx % self.cell_counts[0]
                    elif nx < 0 or nx >= self.cell_counts[0]:
                        continue
                    
                    nid = ny * nx_cells + nx
                    neighbors[count] = np.int64(nid)
                    count += 1
        
        else:  # 3D
            nx_cells = self.cell_counts[0]
            ny_cells = self.cell_counts[1]
            cx = cell_id % nx_cells
            temp = cell_id // nx_cells
            cy = temp % ny_cells
            cz = temp // ny_cells
            
            for dz in range(-cull, cull + 1):
                nz = cz + dz
                if self.periodic:
                    nz = nz % self.cell_counts[2]
                elif nz < 0 or nz >= self.cell_counts[2]:
                    continue
                
                for dy in range(-cull, cull + 1):
                    ny = cy + dy
                    if self.periodic:
                        ny = ny % self.cell_counts[1]
                    elif ny < 0 or ny >= self.cell_counts[1]:
                        continue
                    
                    for dx in range(-cull, cull + 1):
                        nx = cx + dx
                        if self.periodic:
                            nx = nx % self.cell_counts[0]
                        elif nx < 0 or nx >= self.cell_counts[0]:
                            continue
                        
                        nid = (nz * ny_cells + ny) * nx_cells + nx
                        neighbors[count] = np.int64(nid)
                        count += 1
        
        return (neighbors, count)
    
    def _distance(self, idx1: int, idx2: int) -> float:
        """Calculate distance between two particles."""
        if self.ndim == 1:
            return _distance_1d(
                self.positions[idx1, 0], self.positions[idx2, 0],
                self.area_size[0], self.periodic
            )
        elif self.ndim == 2:
            return _distance_2d(
                self.positions[idx1, 0], self.positions[idx1, 1],
                self.positions[idx2, 0], self.positions[idx2, 1],
                self.area_size[0], self.area_size[1], self.periodic
            )
        else:  # 3D
            return _distance_3d(
                self.positions[idx1, 0], self.positions[idx1, 1], self.positions[idx1, 2],
                self.positions[idx2, 0], self.positions[idx2, 1], self.positions[idx2, 2],
                self.area_size[0], self.area_size[1], self.area_size[2], self.periodic
            )
    
    def _record_trace(self, event_type: int, species: int, pos: NDArray[np.float64]) -> None:
        """Record an event in the trace if enabled."""
        if not self.trace_enabled or self.trace_count >= self.trace_capacity:
            return
        
        idx = self.trace_count
        self.trace_type[idx] = event_type
        self.trace_species[idx] = species
        self.trace_time[idx] = self.time
        for d in range(self.ndim):
            self.trace_pos[idx, d] = pos[d]
        self.trace_count += 1
    
    def spawn_particle(self, species: int, x: float, y: float = 0.0, z: float = 0.0) -> bool:
        """Spawn a new particle without recording in trace.
        
        For 1D: spawn_particle(species, x)
        For 2D: spawn_particle(species, x, y)
        For 3D: spawn_particle(species, x, y, z)
        """
        pos_array = np.empty(self.ndim, dtype=np.float64)
        pos_array[0] = x
        if self.ndim >= 2:
            pos_array[1] = y
        if self.ndim >= 3:
            pos_array[2] = z
        return self._spawn_particle_impl(species, pos_array, EVENT_MANUAL_SPAWN, False)
    
    def spawn_particle_traced(self, species: int, x: float, y: float = 0.0, z: float = 0.0) -> bool:
        """Spawn a new particle and record in trace.
        
        For 1D: spawn_particle_traced(species, x)
        For 2D: spawn_particle_traced(species, x, y)
        For 3D: spawn_particle_traced(species, x, y, z)
        """
        pos_array = np.empty(self.ndim, dtype=np.float64)
        pos_array[0] = x
        if self.ndim >= 2:
            pos_array[1] = y
        if self.ndim >= 3:
            pos_array[2] = z
        return self._spawn_particle_impl(species, pos_array, EVENT_MANUAL_SPAWN, True)
    
    def _spawn_particle_impl(
        self, species: int, position: NDArray[np.float64],
        event_type: int, record_trace: bool
    ) -> bool:
        """Internal implementation of particle spawning."""
        if self.population >= self.capacity:
            # Double the total particle capacity
            self._double_particle_capacity()
        
        # Wrap/validate position
        pos = np.empty(self.ndim, dtype=np.float64)
        
        if self.ndim == 1:
            pos[0], valid = _wrap_position_1d(position[0], self.area_size[0], self.periodic)
        elif self.ndim == 2:
            pos[0], pos[1], valid = _wrap_position_2d(
                position[0], position[1],
                self.area_size[0], self.area_size[1], self.periodic
            )
        else:  # 3D
            pos[0], pos[1], pos[2], valid = _wrap_position_3d(
                position[0], position[1], position[2],
                self.area_size[0], self.area_size[1], self.area_size[2], self.periodic
            )
        
        if not valid:
            return False
        
        # Find cell
        cell_id = self._position_to_cell_id(pos)
        cell_slot = self.cell_particle_count[cell_id]
        
        if cell_slot >= self.cell_capacity:
            # Double the cell capacity and retry
            self._double_cell_capacity()
            # cell_slot is still valid, capacity has been doubled
        
        # Allocate particle
        particle_id = int(self.population)
        self.population += 1
        
        # Store particle data
        for d in range(self.ndim):
            self.positions[particle_id, d] = pos[d]
        self.species_id[particle_id] = species
        self.particle_cell[particle_id] = cell_id
        self.particle_slot[particle_id] = cell_slot
        
        # Add to cell
        self.cell_particles[cell_id, cell_slot] = particle_id
        self.cell_particle_count[cell_id] += 1
        self.cell_species_count[cell_id, species] += 1
        
        # Initialize death rate
        base_death = self.d[species]
        self.death_rate[particle_id] = base_death
        
        # Update rates
        base_birth = self.b[species]
        self.total_birth_rate += base_birth
        self.total_death_rate += base_death
        self.cell_birth_rate[cell_id] += base_birth
        self.cell_death_rate[cell_id] += base_death
        self.cell_birth_rate_by_species[cell_id, species] += base_birth
        self.cell_death_rate_by_species[cell_id, species] += base_death
        
        # Handle interactions with neighbors - specialized for 1D for performance
        added_death = 0.0
        if self.ndim == 1:
            # Fast path for 1D
            for other_species in range(self.species_count):
                cutoff_ij = self.cutoff[species, other_species]
                strength_ij = self.dd[species, other_species]
                cutoff_ji = self.cutoff[other_species, species]
                strength_ji = self.dd[other_species, species]
                
                interacts_ij = strength_ij != 0.0 and cutoff_ij > 0.0
                interacts_ji = strength_ji != 0.0 and cutoff_ji > 0.0
                
                if not interacts_ij and not interacts_ji:
                    continue
                
                cull = max(
                    self.cull_range[species, other_species],
                    self.cull_range[other_species, species]
                )
                
                # Direct neighbor iteration for 1D
                cx = cell_id
                for offset in range(-cull, cull + 1):
                    nx = cx + offset
                    if self.periodic:
                        nx = nx % self.cell_counts[0]
                    elif nx < 0 or nx >= self.cell_counts[0]:
                        continue
                    
                    count = self.cell_particle_count[nx]
                    for slot in range(count):
                        other_id = self.cell_particles[nx, slot]
                        if other_id == -1 or other_id == particle_id:
                            continue
                        if self.species_id[other_id] != other_species:
                            continue
                        
                        dist = _distance_1d(
                            self.positions[particle_id, 0],
                            self.positions[other_id, 0],
                            self.area_size[0], self.periodic
                        )
                        
                        if interacts_ij and dist <= cutoff_ij:
                            kernel = _interp_uniform(
                                self.death_x[species, other_species],
                                self.death_y[species, other_species],
                                dist, self.death_inv_dx[species, other_species]
                            )
                            delta = strength_ij * kernel
                            if delta != 0.0:
                                self.death_rate[other_id] += delta
                                self.cell_death_rate_by_species[nx, other_species] += delta
                                self.cell_death_rate[nx] += delta
                                self.total_death_rate += delta
                        
                        if interacts_ji and dist <= cutoff_ji:
                            kernel = _interp_uniform(
                                self.death_x[other_species, species],
                                self.death_y[other_species, species],
                                dist, self.death_inv_dx[other_species, species]
                            )
                            added_death += strength_ji * kernel
        elif self.ndim == 2:
            # Fast path for 2D
            nx_cells = self.cell_counts[0]
            cx = cell_id % nx_cells
            cy = cell_id // nx_cells
            
            for other_species in range(self.species_count):
                cutoff_ij = self.cutoff[species, other_species]
                strength_ij = self.dd[species, other_species]
                cutoff_ji = self.cutoff[other_species, species]
                strength_ji = self.dd[other_species, species]
                
                interacts_ij = strength_ij != 0.0 and cutoff_ij > 0.0
                interacts_ji = strength_ji != 0.0 and cutoff_ji > 0.0
                
                if not interacts_ij and not interacts_ji:
                    continue
                
                cull = max(
                    self.cull_range[species, other_species],
                    self.cull_range[other_species, species]
                )
                
                for dy in range(-cull, cull + 1):
                    ny = cy + dy
                    if self.periodic:
                        ny = ny % self.cell_counts[1]
                    elif ny < 0 or ny >= self.cell_counts[1]:
                        continue
                    
                    for dx in range(-cull, cull + 1):
                        nx = cx + dx
                        if self.periodic:
                            nx = nx % self.cell_counts[0]
                        elif nx < 0 or nx >= self.cell_counts[0]:
                            continue
                        
                        ncell = ny * nx_cells + nx
                        count = self.cell_particle_count[ncell]
                        for slot in range(count):
                            other_id = self.cell_particles[ncell, slot]
                            if other_id == -1 or other_id == particle_id:
                                continue
                            if self.species_id[other_id] != other_species:
                                continue
                            
                            dist = _distance_2d(
                                self.positions[particle_id, 0], self.positions[particle_id, 1],
                                self.positions[other_id, 0], self.positions[other_id, 1],
                                self.area_size[0], self.area_size[1], self.periodic
                            )
                            
                            if interacts_ij and dist <= cutoff_ij:
                                kernel = _interp_uniform(
                                    self.death_x[species, other_species],
                                    self.death_y[species, other_species],
                                    dist, self.death_inv_dx[species, other_species]
                                )
                                delta = strength_ij * kernel
                                if delta != 0.0:
                                    self.death_rate[other_id] += delta
                                    self.cell_death_rate_by_species[ncell, other_species] += delta
                                    self.cell_death_rate[ncell] += delta
                                    self.total_death_rate += delta
                            
                            if interacts_ji and dist <= cutoff_ji:
                                kernel = _interp_uniform(
                                    self.death_x[other_species, species],
                                    self.death_y[other_species, species],
                                    dist, self.death_inv_dx[other_species, species]
                                )
                                added_death += strength_ji * kernel
        else:
            # Fast path for 3D
            nx_cells = self.cell_counts[0]
            ny_cells = self.cell_counts[1]
            cx = cell_id % nx_cells
            temp = cell_id // nx_cells
            cy = temp % ny_cells
            cz = temp // ny_cells
            
            for other_species in range(self.species_count):
                cutoff_ij = self.cutoff[species, other_species]
                strength_ij = self.dd[species, other_species]
                cutoff_ji = self.cutoff[other_species, species]
                strength_ji = self.dd[other_species, species]
                
                interacts_ij = strength_ij != 0.0 and cutoff_ij > 0.0
                interacts_ji = strength_ji != 0.0 and cutoff_ji > 0.0
                
                if not interacts_ij and not interacts_ji:
                    continue
                
                cull = max(
                    self.cull_range[species, other_species],
                    self.cull_range[other_species, species]
                )
                
                for dz in range(-cull, cull + 1):
                    nz = cz + dz
                    if self.periodic:
                        nz = nz % self.cell_counts[2]
                    elif nz < 0 or nz >= self.cell_counts[2]:
                        continue
                    
                    for dy in range(-cull, cull + 1):
                        ny = cy + dy
                        if self.periodic:
                            ny = ny % self.cell_counts[1]
                        elif ny < 0 or ny >= self.cell_counts[1]:
                            continue
                        
                        for dx in range(-cull, cull + 1):
                            nx = cx + dx
                            if self.periodic:
                                nx = nx % self.cell_counts[0]
                            elif nx < 0 or nx >= self.cell_counts[0]:
                                continue
                            
                            ncell = (nz * ny_cells + ny) * nx_cells + nx
                            count = self.cell_particle_count[ncell]
                            for slot in range(count):
                                other_id = self.cell_particles[ncell, slot]
                                if other_id == -1 or other_id == particle_id:
                                    continue
                                if self.species_id[other_id] != other_species:
                                    continue
                                
                                dist = _distance_3d(
                                    self.positions[particle_id, 0], self.positions[particle_id, 1], self.positions[particle_id, 2],
                                    self.positions[other_id, 0], self.positions[other_id, 1], self.positions[other_id, 2],
                                    self.area_size[0], self.area_size[1], self.area_size[2], self.periodic
                                )
                                
                                if interacts_ij and dist <= cutoff_ij:
                                    kernel = _interp_uniform(
                                        self.death_x[species, other_species],
                                        self.death_y[species, other_species],
                                        dist, self.death_inv_dx[species, other_species]
                                    )
                                    delta = strength_ij * kernel
                                    if delta != 0.0:
                                        self.death_rate[other_id] += delta
                                        self.cell_death_rate_by_species[ncell, other_species] += delta
                                        self.cell_death_rate[ncell] += delta
                                        self.total_death_rate += delta
                                
                                if interacts_ji and dist <= cutoff_ji:
                                    kernel = _interp_uniform(
                                        self.death_x[other_species, species],
                                        self.death_y[other_species, species],
                                        dist, self.death_inv_dx[other_species, species]
                                    )
                                    added_death += strength_ji * kernel
        
        if added_death != 0.0:
            self.death_rate[particle_id] += added_death
            self.cell_death_rate_by_species[cell_id, species] += added_death
            self.cell_death_rate[cell_id] += added_death
            self.total_death_rate += added_death
        
        # Record trace if requested
        if record_trace:
            self._record_trace(event_type, species, pos)
        
        return True
    
    def kill_particle_index(self, particle_id: int) -> bool:
        """Kill a particle by its index without recording in trace."""
        return self._kill_particle_impl(particle_id, EVENT_MANUAL_KILL, False)
    
    def kill_particle_index_traced(self, particle_id: int) -> bool:
        """Kill a particle by its index and record in trace."""
        return self._kill_particle_impl(particle_id, EVENT_MANUAL_KILL, True)
    
    def _kill_particle_impl(self, particle_id: int, event_type: int, record_trace: bool) -> bool:
        """Internal implementation of particle killing."""
        if particle_id < 0 or particle_id >= self.population:
            return False
        
        species = self.species_id[particle_id]
        cell_id = self.particle_cell[particle_id]
        slot = self.particle_slot[particle_id]
        
        # Save position for trace
        pos = np.empty(self.ndim, dtype=np.float64)
        for d in range(self.ndim):
            pos[d] = self.positions[particle_id, d]
        
        # Get particle's death rate
        particle_death_rate = self.death_rate[particle_id]
        base_birth = self.b[species]
        
        # Update global rates
        self.total_birth_rate -= base_birth
        self.total_death_rate -= particle_death_rate
        self.cell_birth_rate[cell_id] -= base_birth
        self.cell_death_rate[cell_id] -= particle_death_rate
        self.cell_birth_rate_by_species[cell_id, species] -= base_birth
        self.cell_death_rate_by_species[cell_id, species] -= particle_death_rate
        self.cell_species_count[cell_id, species] -= 1
        
        # Remove interactions with neighbors - specialized for 1D for performance
        if self.ndim == 1:
            # Fast path for 1D
            for other_species in range(self.species_count):
                cutoff_val = self.cutoff[species, other_species]
                strength = self.dd[species, other_species]
                
                if strength == 0.0 or cutoff_val <= 0.0:
                    continue
                
                cull = self.cull_range[species, other_species]
                cx = cell_id
                for offset in range(-cull, cull + 1):
                    nx = cx + offset
                    if self.periodic:
                        nx = nx % self.cell_counts[0]
                    elif nx < 0 or nx >= self.cell_counts[0]:
                        continue
                    
                    count = self.cell_particle_count[nx]
                    for slot_n in range(count):
                        other_id = self.cell_particles[nx, slot_n]
                        if other_id == -1 or other_id == particle_id:
                            continue
                        if self.species_id[other_id] != other_species:
                            continue
                        
                        dist = _distance_1d(
                            self.positions[particle_id, 0],
                            self.positions[other_id, 0],
                            self.area_size[0], self.periodic
                        )
                        if dist <= cutoff_val:
                            kernel = _interp_uniform(
                                self.death_x[species, other_species],
                                self.death_y[species, other_species],
                                dist, self.death_inv_dx[species, other_species]
                            )
                            delta = strength * kernel
                            if delta != 0.0:
                                self.death_rate[other_id] -= delta
                                self.cell_death_rate_by_species[nx, other_species] -= delta
                                self.cell_death_rate[nx] -= delta
                                self.total_death_rate -= delta
        elif self.ndim == 2:
            # Fast path for 2D
            nx_cells = self.cell_counts[0]
            cx = cell_id % nx_cells
            cy = cell_id // nx_cells
            
            for other_species in range(self.species_count):
                cutoff_val = self.cutoff[species, other_species]
                strength = self.dd[species, other_species]
                
                if strength == 0.0 or cutoff_val <= 0.0:
                    continue
                
                cull = self.cull_range[species, other_species]
                
                for dy in range(-cull, cull + 1):
                    ny = cy + dy
                    if self.periodic:
                        ny = ny % self.cell_counts[1]
                    elif ny < 0 or ny >= self.cell_counts[1]:
                        continue
                    
                    for dx in range(-cull, cull + 1):
                        nx = cx + dx
                        if self.periodic:
                            nx = nx % self.cell_counts[0]
                        elif nx < 0 or nx >= self.cell_counts[0]:
                            continue
                        
                        ncell = ny * nx_cells + nx
                        count = self.cell_particle_count[ncell]
                        for slot_n in range(count):
                            other_id = self.cell_particles[ncell, slot_n]
                            if other_id == -1 or other_id == particle_id:
                                continue
                            if self.species_id[other_id] != other_species:
                                continue
                            
                            dist = _distance_2d(
                                self.positions[particle_id, 0], self.positions[particle_id, 1],
                                self.positions[other_id, 0], self.positions[other_id, 1],
                                self.area_size[0], self.area_size[1], self.periodic
                            )
                            if dist <= cutoff_val:
                                kernel = _interp_uniform(
                                    self.death_x[species, other_species],
                                    self.death_y[species, other_species],
                                    dist, self.death_inv_dx[species, other_species]
                                )
                                delta = strength * kernel
                                if delta != 0.0:
                                    self.death_rate[other_id] -= delta
                                    self.cell_death_rate_by_species[ncell, other_species] -= delta
                                    self.cell_death_rate[ncell] -= delta
                                    self.total_death_rate -= delta
        else:
            # Fast path for 3D
            nx_cells = self.cell_counts[0]
            ny_cells = self.cell_counts[1]
            cx = cell_id % nx_cells
            temp = cell_id // nx_cells
            cy = temp % ny_cells
            cz = temp // ny_cells
            
            for other_species in range(self.species_count):
                cutoff_val = self.cutoff[species, other_species]
                strength = self.dd[species, other_species]
                
                if strength == 0.0 or cutoff_val <= 0.0:
                    continue
                
                cull = self.cull_range[species, other_species]
                
                for dz in range(-cull, cull + 1):
                    nz = cz + dz
                    if self.periodic:
                        nz = nz % self.cell_counts[2]
                    elif nz < 0 or nz >= self.cell_counts[2]:
                        continue
                    
                    for dy in range(-cull, cull + 1):
                        ny = cy + dy
                        if self.periodic:
                            ny = ny % self.cell_counts[1]
                        elif ny < 0 or ny >= self.cell_counts[1]:
                            continue
                        
                        for dx in range(-cull, cull + 1):
                            nx = cx + dx
                            if self.periodic:
                                nx = nx % self.cell_counts[0]
                            elif nx < 0 or nx >= self.cell_counts[0]:
                                continue
                            
                            ncell = (nz * ny_cells + ny) * nx_cells + nx
                            count = self.cell_particle_count[ncell]
                            for slot_n in range(count):
                                other_id = self.cell_particles[ncell, slot_n]
                                if other_id == -1 or other_id == particle_id:
                                    continue
                                if self.species_id[other_id] != other_species:
                                    continue
                                
                                dist = _distance_3d(
                                    self.positions[particle_id, 0], self.positions[particle_id, 1], self.positions[particle_id, 2],
                                    self.positions[other_id, 0], self.positions[other_id, 1], self.positions[other_id, 2],
                                    self.area_size[0], self.area_size[1], self.area_size[2], self.periodic
                                )
                                if dist <= cutoff_val:
                                    kernel = _interp_uniform(
                                        self.death_x[species, other_species],
                                        self.death_y[species, other_species],
                                        dist, self.death_inv_dx[species, other_species]
                                    )
                                    delta = strength * kernel
                                    if delta != 0.0:
                                        self.death_rate[other_id] -= delta
                                        self.cell_death_rate_by_species[ncell, other_species] -= delta
                                        self.cell_death_rate[ncell] -= delta
                                        self.total_death_rate -= delta
        
        # Remove from cell
        last_slot = self.cell_particle_count[cell_id] - 1
        last_particle = self.cell_particles[cell_id, last_slot]
        self.cell_particles[cell_id, last_slot] = -1
        self.cell_particle_count[cell_id] = last_slot
        
        if slot != last_slot and last_particle != -1:
            self.cell_particles[cell_id, slot] = last_particle
            self.particle_slot[last_particle] = slot
        
        # Compact particle array
        last_idx = int(self.population) - 1
        self.population -= 1
        
        if particle_id != last_idx:
            # Move last particle to this position
            last_cell = self.particle_cell[last_idx]
            last_slot_id = self.particle_slot[last_idx]
            
            for d in range(self.ndim):
                self.positions[particle_id, d] = self.positions[last_idx, d]
            self.death_rate[particle_id] = self.death_rate[last_idx]
            self.species_id[particle_id] = self.species_id[last_idx]
            self.particle_cell[particle_id] = last_cell
            self.particle_slot[particle_id] = last_slot_id
            
            # Update cell reference
            self.cell_particles[last_cell, last_slot_id] = particle_id
        
        # Clear last particle slot
        for d in range(self.ndim):
            self.positions[last_idx, d] = 0.0
        self.death_rate[last_idx] = 0.0
        self.species_id[last_idx] = -1
        self.particle_cell[last_idx] = -1
        self.particle_slot[last_idx] = -1
        
        # Fix small negative rates due to floating point errors
        if self.total_birth_rate < 0.0 and self.total_birth_rate > -1e-9:
            self.total_birth_rate = 0.0
        if self.total_death_rate < 0.0 and self.total_death_rate > -1e-9:
            self.total_death_rate = 0.0
        
        # Record trace if requested
        if record_trace:
            self._record_trace(event_type, species, pos)
        
        return True
    
    def _sample_parent(self, cell_id: int, species: int) -> int:
        """Sample a random parent from a specific cell and species."""
        count = self.cell_species_count[cell_id, species]
        if count <= 0:
            return -1
        
        target = int(np.random.random() * count)
        cell_count = self.cell_particle_count[cell_id]
        
        for slot in range(cell_count):
            pid = self.cell_particles[cell_id, slot]
            if pid == -1:
                continue
            if self.species_id[pid] == species:
                if target == 0:
                    return pid
                target -= 1
        
        return -1
    
    def _sample_victim(self, cell_id: int, species: int) -> int:
        """Sample a victim weighted by death rate."""
        total = 0.0
        cell_count = self.cell_particle_count[cell_id]
        
        for slot in range(cell_count):
            pid = self.cell_particles[cell_id, slot]
            if pid == -1:
                continue
            if self.species_id[pid] == species:
                total += self.death_rate[pid]
        
        if total <= 0.0:
            return -1
        
        r = np.random.random() * total
        for slot in range(cell_count):
            pid = self.cell_particles[cell_id, slot]
            if pid == -1:
                continue
            if self.species_id[pid] == species:
                r -= self.death_rate[pid]
                if r <= 0.0:
                    return pid
        
        return -1
    
    def attempt_birth_event(self) -> bool:
        """Attempt a birth event."""
        # Sample cell
        cell_id = _sample_weighted(self.cell_birth_rate)
        if cell_id < 0:
            return False
        
        # Sample species
        species = _sample_weighted(self.cell_birth_rate_by_species[cell_id])
        if species < 0:
            return False
        
        # Sample parent
        parent_id = self._sample_parent(cell_id, species)
        if parent_id < 0:
            return False
        
        # Sample dispersal distance
        radius = _interp_uniform(
            self.birth_x[species],
            self.birth_y[species],
            np.random.random(),
            self.birth_inv_dx[species]
        )
        
        # Sample direction and create offspring position
        child_pos = np.empty(self.ndim, dtype=np.float64)
        
        if self.ndim == 1:
            direction = -1.0 if np.random.random() < 0.5 else 1.0
            child_pos[0] = self.positions[parent_id, 0] + direction * radius
        elif self.ndim == 2:
            angle = 2.0 * math.pi * np.random.random()
            child_pos[0] = self.positions[parent_id, 0] + radius * math.cos(angle)
            child_pos[1] = self.positions[parent_id, 1] + radius * math.sin(angle)
        else:  # 3D - sample uniformly on sphere
            theta = 2.0 * math.pi * np.random.random()
            phi = math.acos(2.0 * np.random.random() - 1.0)
            child_pos[0] = self.positions[parent_id, 0] + radius * math.sin(phi) * math.cos(theta)
            child_pos[1] = self.positions[parent_id, 1] + radius * math.sin(phi) * math.sin(theta)
            child_pos[2] = self.positions[parent_id, 2] + radius * math.cos(phi)
        
        return self._spawn_particle_impl(species, child_pos, EVENT_BIRTH, self.trace_enabled)
    
    def attempt_death_event(self) -> bool:
        """Attempt a death event."""
        # Sample cell
        cell_id = _sample_weighted(self.cell_death_rate)
        if cell_id < 0:
            return False
        
        # Sample species
        species = _sample_weighted(self.cell_death_rate_by_species[cell_id])
        if species < 0:
            return False
        
        # Sample victim
        victim_id = self._sample_victim(cell_id, species)
        if victim_id < 0:
            return False
        
        return self._kill_particle_impl(victim_id, EVENT_DEATH, self.trace_enabled)
    
    def run_events(self, max_events: int) -> int:
        """Run a specified number of events."""
        if max_events <= 0:
            return 0
        
        performed = 0
        for _ in range(max_events):
            total_rate = self.total_birth_rate + self.total_death_rate
            if total_rate <= 1e-12:
                break
            
            # Sample time to next event
            dt = -math.log(np.random.random()) / total_rate
            self.time += dt
            
            # Decide event type
            if np.random.random() * total_rate < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
                elif self.capacity_reached:
                    break
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        
        return performed
    
    def run_until_time(self, duration: float) -> int:
        """Run simulation until a specific time duration."""
        if duration <= 0.0:
            return 0
        
        target_time = self.time + duration
        performed = 0
        
        while self.time < target_time:
            total_rate = self.total_birth_rate + self.total_death_rate
            if total_rate <= 1e-12:
                break
            
            # Sample time to next event
            dt = -math.log(np.random.random()) / total_rate
            
            if self.time + dt > target_time:
                self.time = target_time
                break
            
            self.time += dt
            
            # Decide event type
            if np.random.random() * total_rate < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
                elif self.capacity_reached:
                    break
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        
        return performed
    
    def make_event(self) -> bool:
        """Make a single event."""
        return self.run_events(1) > 0
    
    def reached_capacity(self) -> bool:
        """Check if capacity has been reached."""
        return self.capacity_reached
    
    def current_population(self) -> int:
        """Get current population count."""
        return int(self.population)
    
    def current_time(self) -> float:
        """Get current simulation time."""
        return self.time
    
    def current_event_count(self) -> int:
        """Get total event count."""
        return int(self.event_count)
    
    def reseed(self, seed: int) -> None:
        """Reseed the random number generator."""
        np.random.seed(int(seed))
        self.seed = np.int64(seed)
    
    def get_species_counts(self) -> NDArray[np.int32]:
        """Get the count of particles for each species."""
        counts = np.zeros(self.species_count, dtype=np.int32)
        total = int(self.population)
        for idx in range(total):
            species = self.species_id[idx]
            if species >= 0:
                counts[species] += 1
        return counts


# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_cell_counts(
    ndim: int,
    area_size: NDArray[np.float64],
    cutoff: NDArray[np.float64],
    min_cells_per_dim: int = 10
) -> NDArray[np.int32]:
    """
    Automatically calculate cell counts based on cutoff radius.
    Uses maximum cutoff to determine cell size, ensuring at least min_cells_per_dim.
    """
    max_cutoff = np.max(cutoff)
    if max_cutoff <= 0.0:
        max_cutoff = 1.0
    
    cell_counts = np.empty(ndim, dtype=np.int32)
    for d in range(ndim):
        # Divide area size by cutoff radius to get number of cells
        cells = int(area_size[d] / max_cutoff)
        # Ensure minimum
        if cells < min_cells_per_dim:
            cells = min_cells_per_dim
        cell_counts[d] = cells
    
    return cell_counts


def _estimate_capacity(
    species_count: int,
    total_cells: int,
    initial_population: Sequence[Sequence[float]] | None
) -> int:
    """Estimate total capacity needed for particles."""
    if initial_population is not None:
        estimated = 0
        for coords in initial_population:
            estimated += len(coords) if hasattr(coords, '__len__') else 1
        if estimated > 0:
            return max(estimated * 4, 10000)
    
    # Default estimate
    return max(species_count * total_cells * 10, 10000)


def _estimate_cell_capacity(total_capacity: int, total_cells: int) -> int:
    """Estimate per-cell capacity."""
    per_cell = total_capacity // max(1, total_cells)
    # Add buffer
    return max(32, int(per_cell * 2))


# ============================================================================
# Factory Functions
# ============================================================================

def make_ssa_state_1d(
    M: int,
    area_len: float,
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    dd_matrix: Sequence[Sequence[float]] | NDArray[np.float64],
    birth_x: NDArray[np.float64],
    birth_y: NDArray[np.float64],
    death_x: NDArray[np.float64],
    death_y: NDArray[np.float64],
    cutoffs: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    cell_count: int | None = None,
    cell_capacity: int | None = None,
    is_periodic: bool = False,
    seed: int | None = None,
    initial_population: Sequence[Sequence[float]] | None = None,
    trace_enabled: bool = False,
    trace_capacity: int = 10000,
) -> SSAState:
    """Create a 1D SSA state."""
    species_count = int(M)
    
    # Convert parameters to arrays
    b_arr = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d_arr = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd_arr = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    cutoffs_arr = np.ascontiguousarray(cutoffs, dtype=np.float64)
    birth_x_arr = np.ascontiguousarray(birth_x, dtype=np.float64)
    birth_y_arr = np.ascontiguousarray(birth_y, dtype=np.float64)
    death_x_arr = np.ascontiguousarray(death_x, dtype=np.float64)
    death_y_arr = np.ascontiguousarray(death_y, dtype=np.float64)
    
    area_size = np.array([float(area_len)], dtype=np.float64)
    
    # Auto-calculate cell count if not provided
    if cell_count is None:
        cell_counts = _calculate_cell_counts(1, area_size, cutoffs_arr)
    else:
        cell_counts = np.array([int(cell_count)], dtype=np.int32)
    
    total_cells = int(np.prod(cell_counts))
    
    # Estimate capacities
    if cell_capacity is None:
        total_capacity = _estimate_capacity(species_count, total_cells, initial_population)
        per_cell_capacity = _estimate_cell_capacity(total_capacity, total_cells)
    else:
        per_cell_capacity = int(cell_capacity)
        # Use cell_capacity to determine total capacity
        total_capacity = total_cells * per_cell_capacity
    
    seed_val = np.int64(seed if seed is not None else -1)
    
    state = SSAState(
        np.int32(1),
        np.int32(species_count),
        area_size,
        cell_counts,
        is_periodic,
        np.int64(total_capacity),
        np.int32(per_cell_capacity),
        seed_val,
        b_arr, d_arr, dd_arr, cutoffs_arr,
        birth_x_arr, birth_y_arr,
        death_x_arr, death_y_arr,
        trace_enabled,
        np.int64(trace_capacity),
    )
    
    # Initialize population
    if initial_population is not None:
        for species_id, coords in enumerate(initial_population):
            arr = np.asarray(coords, dtype=np.float64).reshape(-1)
            for pos_val in arr:
                if not state.spawn_particle(species_id, float(pos_val)):
                    raise RuntimeError("Initial population exceeds capacity")
    
    return state


def make_ssa_state_2d(
    M: int,
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    dd_matrix: Sequence[Sequence[float]] | NDArray[np.float64],
    birth_x: NDArray[np.float64],
    birth_y: NDArray[np.float64],
    death_x: NDArray[np.float64],
    death_y: NDArray[np.float64],
    cutoffs: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    area_x: float | None = None,
    area_y: float | None = None,
    area_len: NDArray[np.float64] | Sequence[float] | None = None,
    cell_counts: tuple[int, int] | None = None,
    cell_count: NDArray[np.int32] | Sequence[int] | None = None,
    cell_capacity: int | None = None,
    is_periodic: bool = False,
    seed: int | None = None,
    initial_population: Sequence[Sequence[tuple[float, float]]] | None = None,
    trace_enabled: bool = False,
    trace_capacity: int = 10000,
) -> SSAState:
    """Create a 2D SSA state. Supports both (area_x, area_y) and area_len parameters."""
    species_count = int(M)
    
    # Handle area size parameters
    if area_len is not None:
        area_arr = np.asarray(area_len, dtype=np.float64)
        area_size = np.array([float(area_arr[0]), float(area_arr[1])], dtype=np.float64)
    elif area_x is not None and area_y is not None:
        area_size = np.array([float(area_x), float(area_y)], dtype=np.float64)
    else:
        raise ValueError("Must provide either area_len or (area_x, area_y)")
    
    # Convert parameters
    b_arr = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d_arr = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd_arr = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    cutoffs_arr = np.ascontiguousarray(cutoffs, dtype=np.float64)
    birth_x_arr = np.ascontiguousarray(birth_x, dtype=np.float64)
    birth_y_arr = np.ascontiguousarray(birth_y, dtype=np.float64)
    death_x_arr = np.ascontiguousarray(death_x, dtype=np.float64)
    death_y_arr = np.ascontiguousarray(death_y, dtype=np.float64)
    
    # Handle cell count parameters
    if cell_count is not None:
        cell_arr = np.asarray(cell_count, dtype=np.int32)
        cell_counts_arr = np.array([int(cell_arr[0]), int(cell_arr[1])], dtype=np.int32)
    elif cell_counts is not None:
        cell_counts_arr = np.array([int(cell_counts[0]), int(cell_counts[1])], dtype=np.int32)
    else:
        cell_counts_arr = _calculate_cell_counts(2, area_size, cutoffs_arr)
    
    total_cells = int(np.prod(cell_counts_arr))
    
    # Estimate capacities
    total_capacity = _estimate_capacity(species_count, total_cells, initial_population)
    if cell_capacity is None:
        per_cell_capacity = _estimate_cell_capacity(total_capacity, total_cells)
    else:
        per_cell_capacity = int(cell_capacity)
    
    seed_val = np.int64(seed if seed is not None else -1)
    
    state = SSAState(
        np.int32(2),
        np.int32(species_count),
        area_size,
        cell_counts_arr,
        is_periodic,
        np.int64(total_capacity),
        np.int32(per_cell_capacity),
        seed_val,
        b_arr, d_arr, dd_arr, cutoffs_arr,
        birth_x_arr, birth_y_arr,
        death_x_arr, death_y_arr,
        trace_enabled,
        np.int64(trace_capacity),
    )
    
    # Initialize population
    if initial_population is not None:
        for species_id, coords in enumerate(initial_population):
            for coord in coords:
                pos = np.array([float(coord[0]), float(coord[1])], dtype=np.float64)
                if not state.spawn_particle(species_id, pos):
                    raise RuntimeError("Initial population exceeds capacity")
    
    return state


def make_ssa_state_3d(
    M: int,
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    dd_matrix: Sequence[Sequence[float]] | NDArray[np.float64],
    birth_x: NDArray[np.float64],
    birth_y: NDArray[np.float64],
    death_x: NDArray[np.float64],
    death_y: NDArray[np.float64],
    cutoffs: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    area_x: float | None = None,
    area_y: float | None = None,
    area_z: float | None = None,
    area_len: NDArray[np.float64] | Sequence[float] | None = None,
    cell_counts: tuple[int, int, int] | None = None,
    cell_count: NDArray[np.int32] | Sequence[int] | None = None,
    cell_capacity: int | None = None,
    is_periodic: bool = False,
    seed: int | None = None,
    initial_population: Sequence[Sequence[tuple[float, float, float]]] | None = None,
    trace_enabled: bool = False,
    trace_capacity: int = 10000,
) -> SSAState:
    """Create a 3D SSA state. Supports both (area_x, area_y, area_z) and area_len parameters."""
    species_count = int(M)
    
    # Handle area size parameters
    if area_len is not None:
        area_arr = np.asarray(area_len, dtype=np.float64)
        area_size = np.array([float(area_arr[0]), float(area_arr[1]), float(area_arr[2])], dtype=np.float64)
    elif area_x is not None and area_y is not None and area_z is not None:
        area_size = np.array([float(area_x), float(area_y), float(area_z)], dtype=np.float64)
    else:
        raise ValueError("Must provide either area_len or (area_x, area_y, area_z)")
    
    # Convert parameters
    b_arr = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d_arr = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd_arr = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    cutoffs_arr = np.ascontiguousarray(cutoffs, dtype=np.float64)
    birth_x_arr = np.ascontiguousarray(birth_x, dtype=np.float64)
    birth_y_arr = np.ascontiguousarray(birth_y, dtype=np.float64)
    death_x_arr = np.ascontiguousarray(death_x, dtype=np.float64)
    death_y_arr = np.ascontiguousarray(death_y, dtype=np.float64)
    
    # Handle cell count parameters
    if cell_count is not None:
        cell_arr = np.asarray(cell_count, dtype=np.int32)
        cell_counts_arr = np.array([int(cell_arr[0]), int(cell_arr[1]), int(cell_arr[2])], dtype=np.int32)
    elif cell_counts is not None:
        cell_counts_arr = np.array([int(cell_counts[0]), int(cell_counts[1]), int(cell_counts[2])], dtype=np.int32)
    else:
        cell_counts_arr = _calculate_cell_counts(3, area_size, cutoffs_arr)
    
    total_cells = int(np.prod(cell_counts_arr))
    
    # Estimate capacities
    total_capacity = _estimate_capacity(species_count, total_cells, initial_population)
    if cell_capacity is None:
        per_cell_capacity = _estimate_cell_capacity(total_capacity, total_cells)
    else:
        per_cell_capacity = int(cell_capacity)
    
    seed_val = np.int64(seed if seed is not None else -1)
    
    state = SSAState(
        np.int32(3),
        np.int32(species_count),
        area_size,
        cell_counts_arr,
        is_periodic,
        np.int64(total_capacity),
        np.int32(per_cell_capacity),
        seed_val,
        b_arr, d_arr, dd_arr, cutoffs_arr,
        birth_x_arr, birth_y_arr,
        death_x_arr, death_y_arr,
        trace_enabled,
        np.int64(trace_capacity),
    )
    
    # Initialize population
    if initial_population is not None:
        for species_id, coords in enumerate(initial_population):
            for coord in coords:
                pos = np.array([float(coord[0]), float(coord[1]), float(coord[2])], dtype=np.float64)
                if not state.spawn_particle(species_id, pos):
                    raise RuntimeError("Initial population exceeds capacity")
    
    return state


def get_all_particle_coords(state: SSAState) -> list[NDArray[np.float64]]:
    """Get particle coordinates for each species."""
    species_total = int(state.species_count)
    coords = []
    total = int(state.population)
    
    for species_id in range(species_total):
        collected = []
        for idx in range(total):
            if state.species_id[idx] == species_id:
                pos = np.empty(state.ndim, dtype=np.float64)
                for d in range(state.ndim):
                    pos[d] = state.positions[idx, d]
                collected.append(pos)
        
        if len(collected) > 0:
            coords.append(np.array(collected, dtype=np.float64))
        else:
            coords.append(np.empty((0, state.ndim), dtype=np.float64))
    
    return coords


def get_all_particle_death_rates(state: SSAState) -> list[NDArray[np.float64]]:
    """Get death rates for particles of each species."""
    species_total = int(state.species_count)
    result = []
    total = int(state.population)
    
    for species_id in range(species_total):
        collected = []
        for idx in range(total):
            if state.species_id[idx] == species_id:
                collected.append(state.death_rate[idx])
        result.append(np.asarray(collected, dtype=np.float64))
    
    return result


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
