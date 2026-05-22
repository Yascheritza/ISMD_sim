#pragma once

/**
 * @file SpatialBirthDeath.h
 * @brief Header for a spatial birth-death point process simulator (refactored).
 *
 * This simulator models spatially explicit population dynamics
 * with the ability to spawn and kill individuals in a grid of cells.
 * It supports:
 *  - One or more species
 *  - User-defined birth kernels and competition kernels
 *  - Periodic or non-periodic boundaries
 *  - Directed inter-species interactions (dd[s1][s2] can differ from dd[s2][s1])
 *
 * @date 2025-01-20
 */

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>

/**
 * @brief A helper that performs linear interpolation on tabulated (x,y) data.
 *
 * @param xdat The x-values of the tabulated data
 * @param ydat The y-values of the tabulated data
 * @param x The x-value at which to interpolate
 * @return The interpolated y-value
 */
double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat,
                         double x);

/**
 * @brief Iterates over all neighbor cell indices within the specified range around a center cell.
 *
 * This function systematically visits all cells within a hypercubic region centered at the given
 * cell index. For each neighbor cell, it calls the provided callback function with the neighbor's
 * index. The range parameter determines how far to search in each dimension.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3).
 * @tparam FUNC A callable type that accepts a cell index array, e.g., `[](const std::array<int,DIM>
 * &nIdx){ ... }`.
 *
 * @param centerIdx The center cell index around which to search.
 * @param range The maximum offset in each dimension to search (search extends from centerIdx-range
 * to centerIdx+range).
 * @param callback The function to invoke for each neighbor cell index.
 */
template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range,
                  FUNC &&callback);

/**
 * @brief Compute Euclidean distance between two points a,b in DIM dimensions,
 *        with optional periodic wrapping.
 *
 * For each dimension d, if periodic is true and the difference is larger than
 * half the domain length, we wrap it accordingly.
 *
 * @param a The first point.
 * @param b The second point.
 * @param length The domain size along each dimension.
 * @param periodic Whether to apply periodic wrapping.
 * @return The Euclidean distance between a and b.
 */
template <int DIM>
double distancePeriodic(const std::array<double, DIM> &a, const std::array<double, DIM> &b,
                        const std::array<double, DIM> &length, bool periodic);

/**
 * @brief A Cell stores data for multiple species within one grid cell.
 *
 * For each species s:
 *  - coords[s][i] is the i-th particle's coordinates (in real space)
 *  - deathRates[s][i] is the per-particle death rate
 *  - population[s] is the total count of that species in this cell
 *
 * Cached sums:
 *  - cellBirthRateBySpecies[s] - Sum of birth rates for species s in this cell
 *  - cellDeathRateBySpecies[s] - Sum of death rates for species s in this cell
 *  - cellBirthRate - Sum of all species' birth rates in this cell
 *  - cellDeathRate - Sum of all species' death rates in this cell
 */
template <int DIM>
struct Cell {
    /// coords[s]: vector of positions of species s
    std::vector<std::vector<std::array<double, DIM>>> coords;
    /// deathRates[s]: vector of per-particle death rates for species s
    std::vector<std::vector<double>> deathRates;
    /// population[s]: number of individuals of species s
    std::vector<int> population;

    /// Per-species birth and death rates (sums within this cell)
    std::vector<double> cellBirthRateBySpecies;
    std::vector<double> cellDeathRateBySpecies;

    /// Sum of all species' birth rates in this cell
    double cellBirthRate = 0.0;
    /// Sum of all species' death rates in this cell
    double cellDeathRate = 0.0;

    Cell() = default;

    /// @brief Allocate data structures for M species
    /// @param M Number of species to allocate for
    void initSpecies(int M) {
        coords.resize(M);
        deathRates.resize(M);
        population.resize(M, 0);
        cellBirthRateBySpecies.resize(M, 0.0);
        cellDeathRateBySpecies.resize(M, 0.0);
    }
};

/**
 * @brief The main simulation Grid. Partitions the domain into cells.
 *
 * - DIM: the dimension (1,2,3)
 * - M: number of species
 * - b[s], d[s]: per-species baseline birth/death rates
 * - dd[s1][s2]: pairwise interaction magnitude
 * - birth_x[s], birth_y[s], death_x[s1][s2], death_y[s1][s2]: kernels
 * - cutoff[s1][s2]: max distance for interactions
 * - cull[s1][s2]: how many neighboring cells to search in each dimension
 * - cells: the actual grid of size total_num_cells
 * - total_population, total_birth_rate, total_death_rate: global sums
 *
 * The simulation runs by repeatedly calling make_event(), which picks either
 * a birth or a death event according to the ratio of total birth vs death rates.
 *
 * **Refactored** to use:
 *  - spawn_at(...)  / kill_at(...): low-level add/remove of particles
 *  - placePopulation(...): loops over initial coords, calls spawn_at
 *  - spawn_random(...) / kill_random(...): picks random event victims or parents
 */
template <int DIM>
class Grid {
public:
    /// Number of species
    int M;

    /// Physical length of the domain in each dimension
    std::array<double, DIM> area_length;

    /// Number of cells along each dimension
    std::array<int, DIM> cell_count;

    /// If true, boundaries are periodic
    bool periodic;

    /// Per-species baseline birth and death rates
    std::vector<double> b;  ///< b[s]
    std::vector<double> d;  ///< d[s]

    /// Pairwise competition magnitudes: dd[s1][s2]
    std::vector<std::vector<double>> dd;

    /// Birth (dispersal) kernel data
    /// birth_x[s] and birth_y[s] for species s
    std::vector<std::vector<double>> birth_x;
    std::vector<std::vector<double>> birth_y;

    /// Death (competition) kernel data: death_x[s1][s2], death_y[s1][s2]
    std::vector<std::vector<std::vector<double>>> death_x;
    std::vector<std::vector<std::vector<double>>> death_y;

    /// cutoff[s1][s2]: maximum distance for s2->s1 interactions
    std::vector<std::vector<double>> cutoff;

    /// cull[s1][s2][dim]: how many cells to check in +/- directions for neighbor search
    std::vector<std::vector<std::array<int, DIM>>> cull;

    /// The grid cells
    std::vector<Cell<DIM>> cells;

    /// Total number of cells = product of cell_count[dim]
    int total_num_cells;

    /// Global sums for discrete event selection
    double total_birth_rate{0.0};
    double total_death_rate{0.0};

    /// Total count of all species
    int total_population{0};

    /// Random number generator
    std::mt19937 rng;

    /// Simulation time
    double time{0.0};

    /// Total number of events processed
    int event_count{0};

    /// Time point at which the simulation started, for real-time limiting
    std::chrono::system_clock::time_point init_time;

    /// Allowed real-time (in seconds)
    double realtime_limit;

    /// Flag indicating if the real-time limit was reached
    bool realtime_limit_reached{false};

    /**
     * @brief Main constructor.
     *
     * @param M_         Number of species
     * @param areaLen    Domain sizes in each dimension
     * @param cellCount_ Number of cells in each dimension
     * @param isPeriodic If true, domain wraps around boundaries
     * @param birthRates Baseline birth rates for each species (vector of length M)
     * @param deathRates Baseline death rates for each species (vector of length M)
     * @param ddMatrix   Flattened MxM pairwise interaction magnitudes (row-major order)
     * @param birthX     Birth kernel x-values for each species (quantiles in [0,1])
     * @param birthY     Birth kernel y-values for each species (corresponding radii)
     * @param deathX_    Death kernel x-values for each species pair (distances)
     * @param deathY_    Death kernel y-values for each species pair (kernel values)
     * @param cutoffs    Flattened MxM cutoff distances for interactions
     * @param seed       Random number generator seed
     * @param rtimeLimit Real-time limit in seconds for simulation runs
     */
    Grid(int M_, const std::array<double, DIM> &areaLen, const std::array<int, DIM> &cellCount_,
         bool isPeriodic, const std::vector<double> &birthRates,
         const std::vector<double> &deathRates, const std::vector<double> &ddMatrix,
         const std::vector<std::vector<double>> &birthX,
         const std::vector<std::vector<double>> &birthY,
         const std::vector<std::vector<std::vector<double>>> &deathX_,
         const std::vector<std::vector<std::vector<double>>> &deathY_,
         const std::vector<double> &cutoffs, int seed, double rtimeLimit);

    // --- Basic utilities for indexing cells ---
    /**
     * @brief Converts a multi-dimensional cell index to a flat index
     * @param idx Multi-dimensional cell index
     * @return Flattened one-dimensional index
     */
    int flattenIdx(const std::array<int, DIM> &idx) const;

    /**
     * @brief Converts a flat index to a multi-dimensional cell index
     * @param cellIndex Flattened one-dimensional index
     * @return Multi-dimensional cell index
     */
    std::array<int, DIM> unflattenIdx(int cellIndex) const;

    /**
     * @brief Wraps or clamps an index in a specific dimension
     * @param i The index to wrap/clamp
     * @param dim The dimension (0 to DIM-1)
     * @return The wrapped/clamped index
     */
    int wrapIndex(int i, int dim) const;

    /**
     * @brief Checks if a multi-dimensional index is within the domain
     * @param idx Multi-dimensional index to check
     * @return true if index is within domain, false otherwise
     */
    bool inDomain(const std::array<int, DIM> &idx) const;

    /**
     * @brief Gets a reference to the cell at the given raw index (handles wrapping)
     * @param raw Raw multi-dimensional index (may be outside domain if periodic)
     * @return Reference to the cell
     */
    Cell<DIM> &cellAt(const std::array<int, DIM> &raw);

    // --- Kernel evaluation helpers ---
    /**
     * @brief Evaluates the birth kernel for a species at a given quantile
     * @param s Species index
     * @param x Quantile value in [0,1]
     * @return The radius corresponding to the quantile
     */
    double evalBirthKernel(int s, double x) const;

    /**
     * @brief Evaluates the death kernel for a species pair at a given distance
     * @param s1 First species index
     * @param s2 Second species index
     * @param dist Distance between individuals
     * @return The kernel value at the given distance
     */
    double evalDeathKernel(int s1, int s2, double dist) const;

    /**
     * @brief Create a random unit vector in DIM dimensions.
     *        (In 1D, returns either +1 or -1).
     *
     * @param rng Random number generator
     * @return A random unit vector
     */
    std::array<double, DIM> randomUnitVector(std::mt19937 &rng);

    // ------------------------------------------------------------------
    // Refactored interface: direct spawn/kill
    // ------------------------------------------------------------------

    /**
     * @brief Place a new particle of species s at position inPos (wrapping or discarding
     *        if outside domain and periodic==true or false). Update local and global rates.
     *
     * @param s Species index
     * @param inPos The desired real-space position
     */
    void spawn_at(int s, const std::array<double, DIM> &inPos);

    /**
     * @brief Remove exactly one particle of species s in cell cIdx at the specified index.
     *        Updates local and global rates.
     *
     * @param s Species index
     * @param cIdx The cell index array
     * @param victimIdx The index of the particle to remove within the cell's species array
     */
    void kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx);

    /**
     * @brief Removes the interactions contributed by a single particle
     *        (sVictim, victimIdx) in cell cIdx.
     *
     * For each neighbor cell (within cull[sVictim][s2]), we subtract i->j
     * and j->i interactions from occupant j in species s2.
     *
     * @param cIdx The cell index array containing the victim particle
     * @param sVictim Species index of the victim particle
     * @param victimIdx Index of the victim particle within the cell's species array
     */
    void removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx);

    /**
     * @brief Loop over a list of positions for each species and call spawn_at.
     *        Useful to initialize a population or add partial subpopulations.
     *
     * @param initCoords initCoords[s] is a vector of positions for species s.
     */
    void placePopulation(const std::vector<std::vector<std::array<double, DIM>>> &initCoords);

    // ------------------------------------------------------------------
    // Random birth/death events
    // ------------------------------------------------------------------

    /**
     * @brief Perform a random spawn event:
     *   1) pick cell by cellBirthRate
     *   2) pick species by cellBirthRateBySpecies
     *   3) pick a random parent occupant
     *   4) sample a radius from the species' birth kernel, pick random direction
     *   5) call spawn_at(...)
     */
    void spawn_random();

    /**
     * @brief Perform a random kill event:
     *   1) pick cell by cellDeathRate
     *   2) pick species by cellDeathRateBySpecies
     *   3) pick a victim occupant by that species' per-particle deathRates
     *   4) call kill_at(...)
     */
    void kill_random();

    // ------------------------------------------------------------------
    // Core simulation loop
    // ------------------------------------------------------------------

    /**
     * @brief Perform one birth or death event, chosen by ratio of total_birth_rate
     *        to total_death_rate, then sample the waiting time exponentially.
     *
     * Does nothing if total_birth_rate + total_death_rate < 1e-12.
     */
    void make_event();

    /**
     * @brief Run a fixed number of events (birth or death).
     *
     * Terminates early if the real-time limit is reached.
     *
     * @param events Number of events to perform.
     */
    void run_events(int events);

    /**
     * @brief Run the simulation until the specified amount of simulated time has elapsed.
     *
     * Terminates if real-time limit is reached or if total rates vanish.
     *
     * @param duration How much additional simulation time to run.
     */
    void run_for(double duration);

    /**
     * @brief Returns aggregated coordinates for all particles for each species.
     *
     * For each species s (0 <= s < M), returns a vector of particle coordinates.
     * The return type is a vector (per species) of std::array<double, DIM>.
     *
     * @return A vector of vectors containing coordinates for each particle of each species
     */
    std::vector<std::vector<std::array<double, DIM>>> get_all_particle_coords() const;

    /**
     * @brief Returns death rates for all particles for each species.
     *
     * For each species s (0 <= s < M), returns a vector of particle death rates.
     * The return type is a vector (per species) of doubles.
     * The order matches the order of coordinates returned by get_all_particle_coords().
     *
     * @return A vector of vectors containing death rates for each particle of each species
     */
    std::vector<std::vector<double>> get_all_particle_death_rates() const;
};

// Explicit template instantiations
extern template class Grid<1>;
extern template class Grid<2>;
extern template class Grid<3>;

extern template struct Cell<1>;
extern template struct Cell<2>;
extern template struct Cell<3>;
