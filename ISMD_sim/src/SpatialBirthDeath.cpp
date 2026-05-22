/**
 * @file SpatialBirthDeath.cpp
 * @brief Implementation of a spatial birth-death point process simulator.
 *
 * This file contains the implementation of the Grid class template and related functions
 * for simulating spatial birth-death processes in 1, 2, or 3 dimensions.
 *
 * @date 2025-01-20
 */

#include <vector>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <random>
#include "../include/SpatialBirthDeath.h"

double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat,
                         double x) {
    if (x >= xdat.back()) {
        return ydat.back();
    }
    if (x <= xdat.front()) {
        return ydat.front();
    }
    auto i = std::lower_bound(xdat.begin(), xdat.end(), x);  // Nearest-above index
    const size_t k = i - xdat.begin();

    const size_t l = (k > 0) ? k - 1 : 0;  // Nearest-below index

    // Linear interpolation formula
    const double x1 = xdat[l];
    const double x2 = xdat[k];
    const double y1 = ydat[l];
    const double y2 = ydat[k];
    return y1 + ((y2 - y1) * (x - x1) / (x2 - x1));
}

/**
 * @brief Calculates the Euclidean distance between two points in DIM dimensions, with optional
 * periodic wrapping.
 *
 * @tparam DIM The dimension of the space (1, 2, or 3)
 * @param a First point coordinates
 * @param b Second point coordinates
 * @param length Domain lengths in each dimension
 * @param periodic Whether to use periodic boundary conditions
 * @return The Euclidean distance between points a and b
 */
template <int DIM>
double distancePeriodic(const std::array<double, DIM> &a, const std::array<double, DIM> &b,
                        const std::array<double, DIM> &length, bool periodic) {
    double sumSq = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = a[i] - b[i];
        if (periodic) {
            // wrap into [-length/2, length/2]
            if (diff > 0.5 * length[i]) {
                diff -= length[i];
            } else if (diff < -0.5 * length[i]) {
                diff += length[i];
            }
        }
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq);
}

/**
 * @brief Iterates over all neighbor cell indices within the specified range around a center cell.
 *
 * This function systematically visits all cells within a hypercubic region centered at the given
 * cell index. For each neighbor cell, it calls the provided callback function with the neighbor's
 * index. The range parameter determines how far to search in each dimension.
 *
 * Implementation note: This uses dimension-specific code paths for 1D, 2D, and 3D cases
 * for efficiency, rather than a fully generic approach that would work for any dimension.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3).
 * @tparam FUNC A callable type that accepts a cell index array.
 * @param centerIdx The center cell index around which to search.
 * @param range The maximum offset in each dimension to search (search extends from centerIdx-range
 * to centerIdx+range).
 * @param callback The function to invoke for each neighbor cell index.
 */
template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range,
                  FUNC &&callback) {
    // Initialize the neighbor index array
    std::array<int, DIM> neighborIdx;

    // Dimension-specific implementations for efficiency
    if constexpr (DIM == 1) {
        // 1D case: iterate over x-dimension
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];

        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            callback(neighborIdx);
        }
    } else if constexpr (DIM == 2) {
        // 2D case: iterate over x and y dimensions
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];

        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; y++) {
                neighborIdx[1] = y;
                callback(neighborIdx);
            }
        }
    } else if constexpr (DIM == 3) {
        // 3D case: iterate over x, y, and z dimensions
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];
        const int minZ = centerIdx[2] - range[2];
        const int maxZ = centerIdx[2] + range[2];

        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; y++) {
                neighborIdx[1] = y;
                for (int z = minZ; z <= maxZ; z++) {
                    neighborIdx[2] = z;
                    callback(neighborIdx);
                }
            }
        }
    }
}

//============================================================
//  Implementation of Grid<DIM> members
//============================================================

/**
 * @brief Constructor implementation for the simulation grid.
 *
 * Initializes the grid with the specified parameters, setting up the domain,
 * species parameters, and interaction kernels.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 * @param M_ Number of species
 * @param areaLen Domain sizes in each dimension
 * @param cellCount_ Number of cells in each dimension
 * @param isPeriodic Whether to use periodic boundary conditions
 * @param birthRates Baseline birth rates for each species
 * @param deathRates Baseline death rates for each species
 * @param ddMatrix Flattened MxM pairwise interaction magnitudes
 * @param birthX Birth kernel x-values for each species
 * @param birthY Birth kernel y-values for each species
 * @param deathX_ Death kernel x-values for each species pair
 * @param deathY_ Death kernel y-values for each species pair
 * @param cutoffs Flattened MxM cutoff distances for interactions
 * @param seed Random number generator seed
 * @param rtimeLimit Real-time limit in seconds for simulation runs
 */
template <int DIM>
Grid<DIM>::Grid(int M_, const std::array<double, DIM> &areaLen,
                const std::array<int, DIM> &cellCount_, bool isPeriodic,
                const std::vector<double> &birthRates, const std::vector<double> &deathRates,
                const std::vector<double> &ddMatrix, const std::vector<std::vector<double>> &birthX,
                const std::vector<std::vector<double>> &birthY,
                const std::vector<std::vector<std::vector<double>>> &deathX_,
                const std::vector<std::vector<std::vector<double>>> &deathY_,
                const std::vector<double> &cutoffs, int seed, double rtimeLimit)
    : M(M_),
      area_length(areaLen),
      cell_count(cellCount_),
      periodic(isPeriodic),
      rng(seed),
      realtime_limit(rtimeLimit) {
    init_time = std::chrono::system_clock::now();

    // 1) Copy baseline birth & death for each species
    b = birthRates;  // size M
    d = deathRates;  // size M

    // 2) ddMatrix is MxM, stored in row-major as a vector of length M*M
    dd.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        dd[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            dd[s1][s2] = ddMatrix[(s1 * M) + s2];
        }
    }

    // 3) Store birth kernel lookups for each species
    birth_x = birthX;  // birth_x[s] is the x-values (r-values)
    birth_y = birthY;  // birth_y[s] is the y-values for that kernel

    // 4) Store death kernel lookups for (s1, s2)
    death_x.resize(M);
    death_y.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        death_x[s1].resize(M);
        death_y[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            death_x[s1][s2] = deathX_[s1][s2];
            death_y[s1][s2] = deathY_[s1][s2];
        }
    }

    // 5) cutoff[s1][s2]
    cutoff.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        cutoff[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            cutoff[s1][s2] = cutoffs[(s1 * M) + s2];
        }
    }
    // build cull
    cull.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        cull[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            for (int dim = 0; dim < DIM; dim++) {
                const double cellSize = area_length[dim] / cell_count[dim];
                const int needed = (int)std::ceil(cutoff[s1][s2] / cellSize);
                // at least 3 cells
                cull[s1][s2][dim] = std::max(needed, 3);
            }
        }
    }

    // 6) total_num_cells
    {
        int prod = 1;
        for (int dim = 0; dim < DIM; dim++) {
            prod *= cell_count[dim];
        }
        total_num_cells = prod;
    }

    // 7) Allocate cells
    cells.resize(total_num_cells);
    for (auto &c : cells) {
        c.initSpecies(M);
    }
}

template <int DIM>
int Grid<DIM>::flattenIdx(const std::array<int, DIM> &idx) const {
    int f = 0;
    int mul = 1;
    for (int dim = 0; dim < DIM; dim++) {
        f += idx[dim] * mul;
        mul *= cell_count[dim];
    }
    return f;
}

template <int DIM>
std::array<int, DIM> Grid<DIM>::unflattenIdx(int cellIndex) const {
    std::array<int, DIM> cIdx;
    for (int dim = 0; dim < DIM; dim++) {
        cIdx[dim] = cellIndex % cell_count[dim];
        cellIndex /= cell_count[dim];
    }
    return cIdx;
}

template <int DIM>
int Grid<DIM>::wrapIndex(int i, int dim) const {
    if (!periodic) {
        // clamp to [0, cell_count[dim]-1] if nonperiodic
        // or let caller handle out-of-bounds
        return i;
    }
    const int n = cell_count[dim];
    if (i < 0) {
        i += n;  // simple wrap
    } else if (i >= n) {
        i -= n;
    }
    return i;
}

template <int DIM>
bool Grid<DIM>::inDomain(const std::array<int, DIM> &idx) const {
    for (int dim = 0; dim < DIM; dim++) {
        if (idx[dim] < 0 || idx[dim] >= cell_count[dim]) {
            return false;
        }
    }
    return true;
}

template <int DIM>
Cell<DIM> &Grid<DIM>::cellAt(const std::array<int, DIM> &raw) {
    // wrap or clamp
    std::array<int, DIM> w;
    for (int dim = 0; dim < DIM; dim++) {
        w[dim] = wrapIndex(raw[dim], dim);
    }
    return cells[flattenIdx(w)];
}

template <int DIM>
double Grid<DIM>::evalBirthKernel(int s, double x) const {
    return linearInterpolate(birth_x[s], birth_y[s], x);
}

template <int DIM>
double Grid<DIM>::evalDeathKernel(int s1, int s2, double dist) const {
    // directed: only uses dd[s1][s2], plus kernel in death_x[s1][s2], death_y[s1][s2]
    return linearInterpolate(death_x[s1][s2], death_y[s1][s2], dist);
}

/**
 * @brief Generates a random unit vector in DIM dimensions.
 *
 * In 1D, returns either +1 or -1 with equal probability.
 * In higher dimensions, generates a random direction using a normal distribution.
 *
 * @tparam DIM The dimension of the space (1, 2, or 3)
 * @param rng Random number generator
 * @return A random unit vector
 */
template <int DIM>
std::array<double, DIM> Grid<DIM>::randomUnitVector(std::mt19937 &rng) {
    std::array<double, DIM> dir;
    if constexpr (DIM == 1) {
        // In 1D, pick +1 or -1
        std::uniform_real_distribution<double> u(0.0, 1.0);
        dir[0] = (u(rng) < 0.5) ? -1.0 : 1.0;
    } else {
        // For DIM >= 2, pick from normal distribution, then normalize
        std::normal_distribution<double> gauss(0.0, 1.0);
        double sumSq = 0.0;
        for (int d = 0; d < DIM; d++) {
            const double val = gauss(rng);
            dir[d] = val;
            sumSq += val * val;
        }
        const double inv = 1.0 / std::sqrt(sumSq + 1e-14);
        for (int d = 0; d < DIM; d++) {
            dir[d] *= inv;
        }
    }
    return dir;
}

//---------------------------------------------------------
//     spawn_at
//---------------------------------------------------------
/**
 * @brief Places a new particle of species s at position pos.
 *
 * Respects boundary conditions, then updates cell rates and pairwise interactions.
 * If non-periodic and pos is out of range, the particle is discarded.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 * @param s Species index
 * @param inPos The desired real-space position
 */
template <int DIM>
void Grid<DIM>::spawn_at(int s, const std::array<double, DIM> &inPos) {
    // 1) Possibly handle boundary or discard if non-periodic out-of-bounds
    std::array<double, DIM> pos = inPos;
    for (int d = 0; d < DIM; d++) {
        if (pos[d] < 0.0 || pos[d] > area_length[d]) {
            if (!periodic) {
                // out of domain => do nothing
                return;
            }
            // wrap around
            const double L = area_length[d];
            while (pos[d] < 0.0) {
                pos[d] += L;
            }
            while (pos[d] >= L) {
                pos[d] -= L;
            }
        }
    }

    // 2) Determine which cell
    std::array<int, DIM> cIdx;
    for (int d = 0; d < DIM; d++) {
        int c = (int)std::floor(pos[d] * cell_count[d] / area_length[d]);
        if (c == cell_count[d]) {
            c--;
        }
        cIdx[d] = c;
    }
    Cell<DIM> &cell = cellAt(cIdx);

    // 3) Insert occupant
    cell.coords[s].push_back(pos);
    cell.deathRates[s].push_back(d[s]);  // baseline rate for the new occupant
    cell.population[s]++;
    total_population++;

    // 4) Update cell & total birth/death caches by baseline
    cell.cellBirthRateBySpecies[s] += b[s];
    cell.cellBirthRate += b[s];
    total_birth_rate += b[s];

    cell.cellDeathRateBySpecies[s] += d[s];
    cell.cellDeathRate += d[s];
    total_death_rate += d[s];

    // 5) Add pairwise interactions from this new occupant to others
    //    and from others to this occupant.
    //    "Directed" means we do i->j if dd[sNew][s2],
    //    and j->i if dd[s2][sNew], each with separate kernel.
    auto &posNew = cell.coords[s].back();  // the newly added occupant
    int newIdx = (int)cell.coords[s].size() - 1;

    // For each other species s2
    for (int s2 = 0; s2 < M; s2++) {
        auto cullRange = cull[s][s2];
        // neighbor cells
        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic && !inDomain(nIdx)) {
                return;  // skip out-of-bounds neighbor
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            // loop occupant j in s2
            for (int j = 0; j < (int)neighCell.coords[s2].size(); j++) {
                // skip if it is the same occupant
                if (&neighCell == &cell && s2 == s && j == newIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posNew, pos2, area_length, periodic);
                if (dist <= cutoff[s][s2]) {
                    // i->j
                    const double inter_ij = dd[s][s2] * evalDeathKernel(s, s2, dist);
                    // occupant i is the new one
                    // occupant j is in neighCell
                    // occupant j's death rate is increased if s->s2 interaction is non-zero
                    neighCell.deathRates[s2][j] += inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                    neighCell.cellDeathRate += inter_ij;
                    total_death_rate += inter_ij;
                }
                if (dist <= cutoff[s2][s]) {
                    // j->i
                    const double inter_ji = dd[s2][s] * evalDeathKernel(s2, s, dist);
                    // occupant i's death rate is increased if s2->s interaction is non-zero
                    cell.deathRates[s][newIdx] += inter_ji;
                    cell.cellDeathRateBySpecies[s] += inter_ji;
                    cell.cellDeathRate += inter_ji;
                    total_death_rate += inter_ji;
                }
            }
        });
    }
}

//---------------------------------------------------------
//     kill_at
//---------------------------------------------------------
/**
 * @brief Removes exactly one particle of species s in the cell cIdx at the specified index.
 *
 * Updates pairwise interactions and all rates accordingly.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 * @param s Species index
 * @param cIdx The cell index array
 * @param victimIdx The index of the particle to remove within the cell's species array
 */
template <int DIM>
void Grid<DIM>::kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx) {
    Cell<DIM> &cell = cellAt(cIdx);

    // 1) The occupant's baseline death rate is cell.deathRates[s][victimIdx]
    const double victimRate = cell.deathRates[s][victimIdx];

    // 2) Subtract baseline from cell & total
    cell.population[s]--;
    total_population--;

    cell.cellDeathRateBySpecies[s] -= victimRate;
    cell.cellDeathRate -= victimRate;
    total_death_rate -= victimRate;

    // 3) Subtract baseline birth from cell & total
    cell.cellBirthRateBySpecies[s] -= b[s];
    cell.cellBirthRate -= b[s];
    total_birth_rate -= b[s];

    // 4) Remove pairwise interactions contributed by that occupant
    removeInteractionsOfParticle(cIdx, s, victimIdx);

    // 5) Swap-and-pop from coords[s], deathRates[s]
    const int lastIdx = (int)cell.coords[s].size() - 1;
    if (victimIdx != lastIdx) {
        cell.coords[s][victimIdx] = cell.coords[s][lastIdx];
        cell.deathRates[s][victimIdx] = cell.deathRates[s][lastIdx];
    }
    cell.coords[s].pop_back();
    cell.deathRates[s].pop_back();
}

/**
 * @brief Removes all pairwise interactions contributed by a specific particle.
 *
 * Removes both i->j (the occupant's effect on neighbors) and j->i (neighbors' effect on the
 * occupant) interactions from the death rates.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 * @param cIdx The cell index array containing the victim particle
 * @param sVictim Species index of the victim particle
 * @param victimIdx Index of the victim particle within the cell's species array
 */
template <int DIM>
void Grid<DIM>::removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim,
                                             int victimIdx) {
    Cell<DIM> &victimCell = cellAt(cIdx);
    auto &posVictim = victimCell.coords[sVictim][victimIdx];

    // For each species s2, remove i->j and j->i if within cutoff
    for (int s2 = 0; s2 < M; s2++) {
        // cull range for sVictim->s2
        auto range = cull[sVictim][s2];
        forNeighbors<DIM>(cIdx, range, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            // occupant j in species s2
            for (int j = 0; j < (int)neighCell.coords[s2].size(); j++) {
                // skip the victim itself
                if (&neighCell == &victimCell && s2 == sVictim && j == victimIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length, periodic);

                // i->j means occupant i (victim) kills occupant j if dd[sVictim][s2]
                if (dist <= cutoff[sVictim][s2]) {
                    const double inter_ij = dd[sVictim][s2] * evalDeathKernel(sVictim, s2, dist);
                    neighCell.deathRates[s2][j] -= inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                    neighCell.cellDeathRate -= inter_ij;
                    total_death_rate -= inter_ij;
                }
            }
        });
    }
}

//---------------------------------------------------------
//   placePopulation
//---------------------------------------------------------
/**
 * @brief Places multiple particles for each species at specified coordinates.
 *
 * Convenience function that loops over a given set of coordinates for each species
 * and calls spawn_at(s, pos). Replaces old placeInitialPopulations().
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 * @param initCoords Vector of vectors containing coordinates for each species
 */
template <int DIM>
void Grid<DIM>::placePopulation(
    const std::vector<std::vector<std::array<double, DIM>>> &initCoords) {
    // initCoords[s] is a list of positions for species s
    for (int s = 0; s < M; s++) {
        for (auto &pos : initCoords[s]) {
            spawn_at(s, pos);
        }
    }
}

//---------------------------------------------------------
//   spawn_random
//---------------------------------------------------------
/**
 * @brief Performs a random birth event.
 *
 * 1. Picks a random cell (weighted by cellBirthRate)
 * 2. Picks a random species (weighted by cellBirthRateBySpecies)
 * 3. Picks a random occupant in that cell as the "parent"
 * 4. Draws a radius from the birth kernel for that species
 * 5. Generates a random direction
 * 6. Calls spawn_at(...) with the resulting position
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 */
template <int DIM>
void Grid<DIM>::spawn_random() {
    if (total_birth_rate < 1e-12) {
        // no births possible
        return;
    }
    // 1) pick cell
    std::vector<double> cellRateVec(total_num_cells);
    for (int i = 0; i < total_num_cells; i++) {
        cellRateVec[i] = cells[i].cellBirthRate;
    }
    std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int parentCellIndex = cellDist(rng);
    Cell<DIM> &parentCell = cells[parentCellIndex];

    // 2) pick species
    std::discrete_distribution<int> spDist(parentCell.cellBirthRateBySpecies.begin(),
                                           parentCell.cellBirthRateBySpecies.end());
    const int s = spDist(rng);

    // 3) pick parent occupant index
    const int parentIdx = std::uniform_int_distribution<int>(0, parentCell.population[s] - 1)(rng);
    auto &parentPos = parentCell.coords[s][parentIdx];

    // 4) sample radius from the birth kernel
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    const double radius = evalBirthKernel(s, u);

    // 5) random direction
    auto dir = randomUnitVector(rng);
    for (int d = 0; d < DIM; d++) {
        dir[d] *= radius;
    }

    // 6) childPos
    std::array<double, DIM> childPos;
    for (int d = 0; d < DIM; d++) {
        childPos[d] = parentPos[d] + dir[d];
    }

    // 7) call spawn_at
    spawn_at(s, childPos);
}

//---------------------------------------------------------
//    kill_random
//---------------------------------------------------------
/**
 * @brief Performs a random death event.
 *
 * 1. Picks a random cell (weighted by cellDeathRate)
 * 2. Picks a random species within that cell (weighted by cellDeathRateBySpecies)
 * 3. Picks a random occupant within that species (weighted by per-particle deathRates[s][i])
 * 4. Calls kill_at(...) to remove that occupant
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3)
 */
template <int DIM>
void Grid<DIM>::kill_random() {
    if (total_death_rate < 1e-12) {
        // no deaths possible
        return;
    }
    // 1) pick cell
    std::vector<double> cellRateVec(total_num_cells);
    for (int i = 0; i < total_num_cells; i++) {
        cellRateVec[i] = cells[i].cellDeathRate;
    }
    std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int cellIndex = cellDist(rng);
    Cell<DIM> &cell = cells[cellIndex];

    // 2) pick species
    std::discrete_distribution<int> spDist(cell.cellDeathRateBySpecies.begin(),
                                           cell.cellDeathRateBySpecies.end());
    const int s = spDist(rng);

    if (cell.population[s] == 0) {
        return;  // no occupant
    }

    // 3) pick occupant
    std::discrete_distribution<int> victimDist(cell.deathRates[s].begin(),
                                               cell.deathRates[s].end());
    const int victimIdx = victimDist(rng);

    // 4) build cIdx from cellIndex
    const std::array<int, DIM> cIdx = unflattenIdx(cellIndex);

    // 5) kill_at(s, cIdx, victimIdx)
    kill_at(s, cIdx, victimIdx);
}

//---------------------------------------------------------
//   make_event, run_events, run_for
//---------------------------------------------------------

template <int DIM>
void Grid<DIM>::make_event() {
    const double sumRate = total_birth_rate + total_death_rate;
    if (sumRate < 1e-12) {
        // no event possible
        return;
    }
    event_count++;

    // 1) Advance time by Exp(sumRate)
    std::exponential_distribution<double> expDist(sumRate);
    const double dt = expDist(rng);
    time += dt;

    // 2) Decide birth vs death
    const double r = std::uniform_real_distribution<double>(0.0, sumRate)(rng);
    const bool isBirth = (r < total_birth_rate);

    if (isBirth) {
        spawn_random();
    } else {
        kill_random();
    }
}

template <int DIM>
void Grid<DIM>::run_events(int events) {
    for (int i = 0; i < events; i++) {
        // optionally check real time limit
        if (std::chrono::system_clock::now() >
            init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
    }
}

template <int DIM>
void Grid<DIM>::run_for(double duration) {
    const double endTime = time + duration;
    while (time < endTime) {
        // real-time limit?
        if (std::chrono::system_clock::now() >
            init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
        if (total_birth_rate + total_death_rate < 1e-12) {
            // no more events possible
            return;
        }
    }
}

template <int DIM>
std::vector<std::vector<std::array<double, DIM>>> Grid<DIM>::get_all_particle_coords() const {
    // Prepare one vector per species.
    std::vector<std::vector<std::array<double, DIM>>> result(M);
    // Loop over all cells
    for (const auto &cell : cells) {
        // For each species, add the cell’s coordinates to the species vector.
        for (int s = 0; s < M; ++s) {
            result[s].insert(result[s].end(), cell.coords[s].begin(), cell.coords[s].end());
        }
    }
    return result;
}

template <int DIM>
std::vector<std::vector<double>> Grid<DIM>::get_all_particle_death_rates() const {
    std::vector<std::vector<double>> result(M);
    for (const auto &cell : cells) {
        for (int s = 0; s < M; ++s) {
            result[s].insert(result[s].end(), cell.deathRates[s].begin(), cell.deathRates[s].end());
        }
    }
    return result;
}

template class Grid<1>;
template class Grid<2>;
template class Grid<3>;
