#cython: language_level=3
"""
SpatialBirthDeathWrapper.pyx - Cython wrapper for the C++ spatial birth-death simulator.

This module provides Python classes (PyGrid1, PyGrid2, PyGrid3) that wrap the C++ Grid<DIM>
template class, allowing Python users to create and manipulate spatial birth-death
simulations in 1, 2, or 3 dimensions.
"""

# Import standard Cython / Python definitions
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t

#######################################################
# 1) Declare std::array<T,N> for needed combinations
#######################################################
# These declarations allow Cython to work with C++ std::array types
cdef extern from "<array>" namespace "std" nogil:

    # -- double, dimension=1 --
    cdef cppclass arrayDouble1 "std::array<double, 1>":
        arrayDouble1() except +
        double& operator[](size_t) except +

    # -- double, dimension=2 --
    cdef cppclass arrayDouble2 "std::array<double, 2>":
        arrayDouble2() except +
        double& operator[](size_t) except +

    # -- double, dimension=3 --
    cdef cppclass arrayDouble3 "std::array<double, 3>":
        arrayDouble3() except +
        double& operator[](size_t) except +

    # -- int, dimension=1 --
    cdef cppclass arrayInt1 "std::array<int, 1>":
        arrayInt1() except +
        int& operator[](size_t) except +

    # -- int, dimension=2 --
    cdef cppclass arrayInt2 "std::array<int, 2>":
        arrayInt2() except +
        int& operator[](size_t) except +

    # -- int, dimension=3 --
    cdef cppclass arrayInt3 "std::array<int, 3>":
        arrayInt3() except +
        int& operator[](size_t) except +

#######################################################
# 2) Provide helper functions to convert Python lists to std::array
#######################################################
# These functions convert Python lists to C++ std::array objects

cdef arrayDouble1 pyToStdArrayDouble1(list arr) except *:
    """Convert a Python list with 1 float to std::array<double, 1>"""
    cdef arrayDouble1 result = arrayDouble1()
    result[0] = <double>arr[0]
    return result

cdef arrayDouble2 pyToStdArrayDouble2(list arr) except *:
    """Convert a Python list with 2 floats to std::array<double, 2>"""
    cdef arrayDouble2 result = arrayDouble2()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    return result

cdef arrayDouble3 pyToStdArrayDouble3(list arr) except *:
    """Convert a Python list with 3 floats to std::array<double, 3>"""
    cdef arrayDouble3 result = arrayDouble3()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    result[2] = <double>arr[2]
    return result

cdef arrayInt1 pyToStdArrayInt1(list arr) except *:
    """Convert a Python list with 1 integer to std::array<int, 1>"""
    cdef arrayInt1 result = arrayInt1()
    result[0] = <int>arr[0]
    return result

cdef arrayInt2 pyToStdArrayInt2(list arr) except *:
    """Convert a Python list with 2 integers to std::array<int, 2>"""
    cdef arrayInt2 result = arrayInt2()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    return result

cdef arrayInt3 pyToStdArrayInt3(list arr) except *:
    """Convert a Python list with 3 integers to std::array<int, 3>"""
    cdef arrayInt3 result = arrayInt3()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    result[2] = <int>arr[2]
    return result

#######################################################
# 3) Convert Python lists to std::vector<double> and nested vectors
#######################################################
# These functions convert Python lists to C++ std::vector objects

cdef vector[double] pyListToVectorDouble(object pyList) except *:
    """Convert a Python list of floats to std::vector<double>"""
    cdef int n = len(pyList)
    cdef vector[double] vec
    vec.resize(n)
    cdef int i
    cdef double val
    for i in range(n):
        val = pyList[i]
        vec[i] = val
    return vec

cdef vector[ vector[double] ] pyListOfListToVectorVectorDouble(object pyList) except *:
    """Convert a Python list of lists of floats to std::vector<std::vector<double>>"""
    cdef int outer_size = len(pyList)
    cdef vector[ vector[double] ] result
    result.resize(outer_size)

    cdef int i, inner_size
    cdef object inner_list
    cdef int j
    cdef double val
    for i in range(outer_size):
        inner_list = pyList[i]
        inner_size = len(inner_list)
        result[i].resize(inner_size)
        for j in range(inner_size):
            val = inner_list[j]
            result[i][j] = val
    return result

cdef vector[ vector[ vector[double] ] ] pyListOfListOfListToVector3Double(object pyList) except *:
    """Convert a Python list of lists of lists of floats to std::vector<std::vector<std::vector<double>>>"""
    cdef int s1_count = len(pyList)
    cdef vector[ vector[ vector[double] ] ] out3
    out3.resize(s1_count)

    cdef object second_level
    cdef object third_level
    cdef int s1, s2_count, s2
    for s1 in range(s1_count):
        second_level = pyList[s1]
        s2_count = len(second_level)
        out3[s1].resize(s2_count)
        for s2 in range(s2_count):
            third_level = second_level[s2]
            out3[s1][s2] = pyListToVectorDouble(third_level)
    return out3

#######################################################
# 4) Convert Python coordinate lists for placePopulation(...)
#    (dimension-specific implementations)
#######################################################
# These functions convert Python coordinate lists to C++ vector of vectors of arrays
# for use with the placePopulation method

cdef vector[ vector[ arrayDouble1 ] ] pyToCoordsD1(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 1D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x].
    
    Returns:
        A vector of vectors of arrayDouble1 for use with Grid<1>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble1 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list singlePos
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            singlePos = posList[i]
            result[s][i] = pyToStdArrayDouble1(singlePos)
    return result

cdef vector[ vector[ arrayDouble2 ] ] pyToCoordsD2(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 2D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x, y].
    
    Returns:
        A vector of vectors of arrayDouble2 for use with Grid<2>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble2 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xy
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xy = posList[i]
            result[s][i] = pyToStdArrayDouble2(xy)
    return result

cdef vector[ vector[ arrayDouble3 ] ] pyToCoordsD3(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 3D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x, y, z].
    
    Returns:
        A vector of vectors of arrayDouble3 for use with Grid<3>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble3 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xyz
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xyz = posList[i]
            result[s][i] = pyToStdArrayDouble3(xyz)
    return result

#######################################################
# 5) Expose Cell<DIM> classes (for reading only)
#######################################################
# These declarations allow Cython to access the C++ Cell template classes
cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell1 "Cell<1>":
        vector[ vector[arrayDouble1] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell2 "Cell<2>":
        vector[ vector[arrayDouble2] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell3 "Cell<3>":
        vector[ vector[arrayDouble3] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

#######################################################
# 6) Expose Grid<DIM> classes with their methods
#######################################################
# These declarations allow Cython to access the C++ Grid template classes
cdef extern from "SpatialBirthDeath.h":

    cdef cppclass Grid1 "Grid<1>":
        Grid1(int M_,
              arrayDouble1 areaLen,
              arrayInt1 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        # New approach: placePopulation, spawn_at, kill_at
        void placePopulation(const vector[ vector[ arrayDouble1 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble1 &inPos) except +
        void kill_at(int s, const arrayInt1 &cIdx, int victimIdx) except +

        # The random event methods
        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble1]] get_all_particle_coords() except +
        vector[vector[double]] get_all_particle_death_rates() except +

        # Exposed fields:
        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell1] cells

    cdef cppclass Grid2 "Grid<2>":
        Grid2(int M_,
              arrayDouble2 areaLen,
              arrayInt2 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble2 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble2 &inPos) except +
        void kill_at(int s, const arrayInt2 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble2]] get_all_particle_coords() except +
        vector[vector[double]] get_all_particle_death_rates() except +

        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell2] cells

    cdef cppclass Grid3 "Grid<3>":
        Grid3(int M_,
              arrayDouble3 areaLen,
              arrayInt3 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble3 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble3 &inPos) except +
        void kill_at(int s, const arrayInt3 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble3]] get_all_particle_coords() except +
        vector[vector[double]] get_all_particle_death_rates() except +

        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell3] cells

#######################################################
# 7) Python wrapper classes for the C++ Grid classes
#######################################################
# These classes expose the C++ functionality to Python

#==================== PyGrid1 (wrapper for Grid<1>) ====================#
cdef class PyGrid1:
    cdef Grid1* cpp_grid  # Owned pointer

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [25.0]
                  cellCount,        # Number of cells, e.g. [25]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX_,          # Death kernel x-values (distances)
                  deathY_,          # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds
        """
        Initialize a 1D spatial birth-death simulator.
        
        Parameters:
            M: Number of species
            areaLen: Domain size as a list [length]
            cellCount: Number of cells as a list [num_cells]
            isPeriodic: Whether to use periodic boundary conditions
            birthRates: List of baseline birth rates for each species
            deathRates: List of baseline death rates for each species
            ddMatrix: Flattened MxM matrix of pairwise interaction magnitudes
            birthX: List of lists, where birthX[s] contains quantiles for species s
            birthY: List of lists, where birthY[s] contains radii for species s
            deathX_: 3D list of death kernel x-values (distances)
            deathY_: 3D list of death kernel y-values (kernel values)
            cutoffs: Flattened MxM list of cutoff distances
            seed: Random number generator seed
            rtimeLimit: Real-time limit in seconds for simulation runs
        """
        cdef arrayDouble1 c_areaLen = pyToStdArrayDouble1(areaLen)
        cdef arrayInt1    c_cellCount = pyToStdArrayInt1(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid1(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ Population placement ------
    def placePopulation(self, initCoords):
        """
        Place multiple particles at specified coordinates.
        
        Parameters:
            initCoords: List of lists where initCoords[s] is a list of positions for species s.
                        For 1D, each position is a list [x].
        """
        cdef vector[ vector[arrayDouble1] ] c_init = pyToCoordsD1(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ Direct spawn/kill methods ------
    def spawn_at(self, s, pos):
        """
        Place a new particle of species s at the specified position.
        
        Parameters:
            s: Species index (integer)
            pos: Position as a list [x]
        """
        cdef arrayDouble1 cpos = pyToStdArrayDouble1(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        Remove a particle of species s in the specified cell at the given index.
        
        Parameters:
            s: Species index (integer)
            cell_idx: Cell index as a list [i]
            victimIdx: Index of the particle within the cell's species array
        """
        cdef arrayInt1 cc = pyToStdArrayInt1(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ Random event methods ------
    def spawn_random(self):
        """
        Perform a random birth event.
        
        Picks a random cell weighted by birth rates, then a random species,
        then a random parent, and places a new particle at a distance
        sampled from the birth kernel in a random direction.
        """
        self.cpp_grid.spawn_random()

    def kill_random(self):
        """
        Perform a random death event.
        
        Picks a random cell weighted by death rates, then a random species,
        then a random victim weighted by per-particle death rates, and removes it.
        """
        self.cpp_grid.kill_random()

    def make_event(self):
        """
        Perform one birth or death event and advance simulation time.
        
        The event type (birth or death) is chosen based on the ratio of
        total birth rate to total death rate. The waiting time is sampled
        from an exponential distribution.
        """
        self.cpp_grid.make_event()

    def run_events(self, n):
        """
        Run a fixed number of events.
        
        Parameters:
            n: Number of events to perform
            
        The simulation will terminate early if the real-time limit is reached.
        """
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        """
        Run the simulation for a specified amount of simulated time.
        
        Parameters:
            duration: Amount of simulation time to advance
            
        The simulation will terminate early if the real-time limit is reached
        or if the total rates become negligible.
        """
        self.cpp_grid.run_for(duration)

    # ------ Read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        """
        Get the total number of cells in the grid.
        
        Returns:
            Integer representing the total number of cells
        """
        return self.cpp_grid.total_num_cells

    # ------ Cell data access methods ------
    def get_cell_coords(self, cell_index, species_idx):
        """
        Get the coordinates of all particles of a species in a cell.
        
        Parameters:
            cell_index: Index of the cell
            species_idx: Index of the species
            
        Returns:
            List of x-coordinates for all particles of the specified species in the cell
        """
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble1] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append(coords_vec[i][0])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        """
        Get the death rates of all particles of a species in a cell.
        
        Parameters:
            cell_index: Index of the cell
            species_idx: Index of the species
            
        Returns:
            List of death rates for all particles of the specified species in the cell
        """
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        """
        Get the population counts for each species in a cell.
        
        Parameters:
            cell_index: Index of the cell
            
        Returns:
            List of population counts, one for each species
        """
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        """
        Get the total birth rate in a cell.
        
        Parameters:
            cell_index: Index of the cell
            
        Returns:
            Total birth rate summed over all species in the cell
        """
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        """
        Get the total death rate in a cell.
        
        Parameters:
            cell_index: Index of the cell
            
        Returns:
            Total death rate summed over all species in the cell
        """
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 1D grid, each coordinate is a float.

        Returns:
            A list of lists: one list per species, where each inner list
            contains the x-coordinate (a float) for every particle.
        """
        cdef vector[vector[arrayDouble1]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                # For 1D, each coordinate is stored in a std::array<double,1>
                species_coords.append(c_all[i][j][0])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        """
        Returns aggregated death rates for all particles in the grid,
        grouped by species.

        Returns:
            A list of lists: one list per species, where each inner list
            contains the death rate for every particle of that species.
        """
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out


#==================== PyGrid2 (wrapper for Grid<2>) ====================#
cdef class PyGrid2:
    cdef Grid2* cpp_grid

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [width, height]
                  cellCount,        # Number of cells, e.g. [nx, ny]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX_,          # Death kernel x-values (distances)
                  deathY_,          # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds
        """
        Initialize a 2D spatial birth-death simulator.
        
        Parameters:
            M: Number of species
            areaLen: Domain size as a list [width, height]
            cellCount: Number of cells as a list [nx, ny]
            isPeriodic: Whether to use periodic boundary conditions
            birthRates: List of baseline birth rates for each species
            deathRates: List of baseline death rates for each species
            ddMatrix: Flattened MxM matrix of pairwise interaction magnitudes
            birthX: List of lists, where birthX[s] contains quantiles for species s
            birthY: List of lists, where birthY[s] contains radii for species s
            deathX_: 3D list of death kernel x-values (distances)
            deathY_: 3D list of death kernel y-values (kernel values)
            cutoffs: Flattened MxM list of cutoff distances
            seed: Random number generator seed
            rtimeLimit: Real-time limit in seconds for simulation runs
        """
        cdef arrayDouble2 c_areaLen = pyToStdArrayDouble2(areaLen)
        cdef arrayInt2    c_cellCount = pyToStdArrayInt2(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid2(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ Population placement ------
    def placePopulation(self, initCoords):
        """
        Place multiple particles at specified coordinates.
        
        Parameters:
            initCoords: List of lists where initCoords[s] is a list of positions for species s.
                        For 2D, each position is a list [x, y].
        """
        cdef vector[ vector[arrayDouble2] ] c_init = pyToCoordsD2(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ Direct spawn/kill methods ------
    def spawn_at(self, s, pos):
        """
        Place a new particle of species s at the specified position.
        
        Parameters:
            s: Species index (integer)
            pos: Position as a list [x, y]
        """
        cdef arrayDouble2 cpos = pyToStdArrayDouble2(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        Remove a particle of species s in the specified cell at the given index.
        
        Parameters:
            s: Species index (integer)
            cell_idx: Cell index as a list [ix, iy]
            victimIdx: Index of the particle within the cell's species array
        """
        cdef arrayInt2 cc = pyToStdArrayInt2(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ random events ------
    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

    def make_event(self):
        self.cpp_grid.make_event()

    def run_events(self, n):
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        self.cpp_grid.run_for(duration)

    # ------ read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        return self.cpp_grid.total_num_cells

    # ------ Access cell data ------
    def get_cell_coords(self, cell_index, species_idx):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble2] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0], coords_vec[i][1]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 2D grid, each coordinate is a list [x, y].

        Returns:
            A list of lists: one list per species, where each inner list
            contains the [x, y] coordinates of a particle.
        """
        cdef vector[vector[arrayDouble2]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0], c_all[i][j][1]])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        """
        Returns aggregated death rates for all particles in the grid,
        grouped by species.

        Returns:
            A list of lists: one list per species, where each inner list
            contains the death rate for every particle of that species.
        """
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out

#==================== PyGrid3 (wrapper for Grid<3>) ====================#
cdef class PyGrid3:
    cdef Grid3* cpp_grid

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [x_size, y_size, z_size]
                  cellCount,        # Number of cells, e.g. [nx, ny, nz]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX_,          # Death kernel x-values (distances)
                  deathY_,          # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds
        """
        Initialize a 3D spatial birth-death simulator.
        
        Parameters:
            M: Number of species
            areaLen: Domain size as a list [x_size, y_size, z_size]
            cellCount: Number of cells as a list [nx, ny, nz]
            isPeriodic: Whether to use periodic boundary conditions
            birthRates: List of baseline birth rates for each species
            deathRates: List of baseline death rates for each species
            ddMatrix: Flattened MxM matrix of pairwise interaction magnitudes
            birthX: List of lists, where birthX[s] contains quantiles for species s
            birthY: List of lists, where birthY[s] contains radii for species s
            deathX_: 3D list of death kernel x-values (distances)
            deathY_: 3D list of death kernel y-values (kernel values)
            cutoffs: Flattened MxM list of cutoff distances
            seed: Random number generator seed
            rtimeLimit: Real-time limit in seconds for simulation runs
        """
        cdef arrayDouble3 c_areaLen = pyToStdArrayDouble3(areaLen)
        cdef arrayInt3    c_cellCount = pyToStdArrayInt3(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid3(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ Population placement ------
    def placePopulation(self, initCoords):
        """
        Place multiple particles at specified coordinates.
        
        Parameters:
            initCoords: List of lists where initCoords[s] is a list of positions for species s.
                        For 3D, each position is a list [x, y, z].
        """
        cdef vector[ vector[arrayDouble3] ] c_init = pyToCoordsD3(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ Direct spawn/kill methods ------
    def spawn_at(self, s, pos):
        """
        Place a new particle of species s at the specified position.
        
        Parameters:
            s: Species index (integer)
            pos: Position as a list [x, y, z]
        """
        cdef arrayDouble3 cpos = pyToStdArrayDouble3(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        Remove a particle of species s in the specified cell at the given index.
        
        Parameters:
            s: Species index (integer)
            cell_idx: Cell index as a list [ix, iy, iz]
            victimIdx: Index of the particle within the cell's species array
        """
        cdef arrayInt3 cc = pyToStdArrayInt3(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ random events ------
    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

    def make_event(self):
        self.cpp_grid.make_event()

    def run_events(self, n):
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        self.cpp_grid.run_for(duration)

    # ------ read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        return self.cpp_grid.total_num_cells

    # ------ Access cell data ------
    def get_cell_coords(self, cell_index, species_idx):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble3] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0],
                        coords_vec[i][1],
                        coords_vec[i][2]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 3D grid, each coordinate is a list [x, y, z].

        Returns:
            A list of lists: one list per species, where each inner list
            contains the [x, y, z] coordinates of a particle.
        """
        cdef vector[vector[arrayDouble3]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0],
                                       c_all[i][j][1],
                                       c_all[i][j][2]])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        """
        Returns aggregated death rates for all particles in the grid,
        grouped by species.

        Returns:
            A list of lists: one list per species, where each inner list
            contains the death rate for every particle of that species.
        """
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out
