#ifndef ASTAR_H
#define ASTAR_H

#include <limits>
#include <list>
#include <cstdint>
#include <cmath>

#include "point.h"
#include "direction.h"

namespace simulation {

    enum class ArenaTile : uint8_t {
        Empty = 0,
        Wall = 1,
        Player = 2,
        Goal = 3
    };

}

namespace pathfinding {
    template <std::size_t N, std::size_t M, typename StorageType>
    class Dijikstra2D;

    namespace detail {
        template <typename T, std::size_t ... Is>
        constexpr std::array<T, sizeof...(Is)>
        create_array(T value, std::index_sequence<Is...>) {
            // cast Is to void to remove the warning: unused value
            return {{(static_cast<void>(Is), value)...}};
        }
    }

    template <std::size_t N, typename T>
    constexpr std::array<T, N> create_array(const T& value) {
        return detail::create_array(value, std::make_index_sequence<N>());
    }

    template <std::size_t N, std::size_t M, typename T>
    constexpr std::array<std::array<T, M>, N> create_matrix(const T& value) {
        return create_array<N, std::array<T, M>>(create_array<M, T>(value));
    }
}

namespace pathfinding {

    template <std::size_t N, std::size_t M, typename T>
    using Matrix = std::array<std::array<T, M>, N>;
    template <std::size_t N, std::size_t M>
    using Grid = Matrix<N, M, simulation::ArenaTile>;

    template <std::size_t N, std::size_t M, typename StorageType = simulation::PointStorageType>
    class Dijikstra2D {
        using Node = simulation::Point<StorageType>;
    public:
        virtual ~Dijikstra2D() = default;

        Dijikstra2D(const Grid<N, M> &grid) :
        grid(create_matrix<N, M, simulation::ArenaTile>(simulation::ArenaTile::Empty)),
        distances(create_matrix<N, M, StorageType>(INF)),
        previous(create_matrix<N, M, Node>({-1, -1})),
        lastStart(-1, -1),
        lastEnd(-1, -1){
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != M; ++j) {
                    this->grid[j][i] = grid[j][i];
                }
            }
        }

        // Returns a list of all nodes that are neighbors to the given node
        virtual std::list<Node> neighbors(const Node &point) {
            constexpr std::array<simulation::Direction, 4> directions = {
                simulation::Direction::Up,
                simulation::Direction::Down,
                simulation::Direction::Left,
                simulation::Direction::Right
            };
            std::list<Node> neighbors;
            for (const auto &dir : directions) {
                auto tmp = point + dir; // Get the transformed point
                // Check for wrapping and shift accordingly
                if (tmp.getX() < 0) tmp.setX(N - 1);
                if (tmp.getX() == N) tmp.setX(0);
                // Check if the point in inbounds and a valid neighbor
                if ((tmp.getX() < 0 or tmp.getX() > N ) or
                                            (tmp.getY() < 0 or tmp.getY() > M) or
                                            tile(point) != simulation::ArenaTile::Wall) {
                    neighbors.emplace_back(tmp); // Add the point to the list of neighbors
                }
            }
            return neighbors;
        }
        // Returns a score for the given point based on the target point
        virtual int32_t score(const Node& current, const Node& end) {
            return std::abs(current.getX() - end.getX()) + std::abs(current.getY() - end.getY());
        }
        // Returns the distance between the two given points
        virtual StorageType distance(const Node &, const Node &) {
            return 1;
        }

        // Computes the path for the given start and end nodes
        std::list<Node> path(const Node &start, const Node &end) {
            // Only calculate the path if we don't already have it
            if (start != lastStart or end != lastEnd) {
                // Clear the internal structures for usage in the new path
                for (auto &row : this->distances) {
                    row.fill(INF);
                }
                for (auto &row : this->previous) {
                    row.fill({-1, -1});
                }
                // Compute the path in the internal structure
                _path(start, end, 0);
            }
            // Generate the path from the internal structure
            std::list<Node> path;
            Node current = end;
            while (current != start) {
                path.push_back(current);
                current = previous[current.getY()][current.getX()];
            }
            path.reverse(); // Reverse since we generate the path starting from the end
            return path;
        }
    protected:
        // Recursive implementation of dijikstra's path finding algorithm with score pruning
        StorageType _path(const Node &current, const Node& end, const int32_t currentDist) {
            if (current == end) {
                return currentDist;
            }

            auto currentScore = this->score(current, end);

            auto minDist = INF;
            const auto points = neighbors(current);
            for (const auto &neighbor : points) {
                int32_t newDist = currentDist + 1;
                if (newDist < distances[neighbor.getY()][neighbor.getX()]) {
                    distances[neighbor.getY()][neighbor.getX()] = newDist;
                    previous[neighbor.getY()][neighbor.getX()] = current;  // Store parent point
                    auto newScore = std::abs(neighbor.getX() - end.getX()) + std::abs(neighbor.getY() - end.getY());
                    if (newScore < currentScore)
                        minDist = std::min(minDist, _path(neighbor, end, newDist));
                }
            }
            return minDist;
        }
        // Retrieve a tile from the grid
        simulation::ArenaTile tile(const Node &n) { return this->grid[n.getY()][n.getX()]; }

        constexpr static auto INF = std::numeric_limits<StorageType>::max();
        Grid<N, M> grid;
        Matrix<N, M, StorageType> distances;
        Matrix<N, M, Node> previous;
        Node lastStart;
        Node lastEnd;
    };

}

/*

// Dijkstra algorithm for 2D grid using Eigen::Matrix
template<typename StorageType = simulation::PointStorageType, StorageType Rows, StorageType Cols>
std::vector<simulation::Point<StorageType>> dijkstra(const Eigen::Ref<Eigen::Matrix<simulation::ArenaTile, Rows, Cols, Eigen::RowMajor>> &grid, const simulation::Point<StorageType> &start, const simulation::Point<StorageType> &end) {
    using GridPoint = simulation::Point<StorageType>;
    using DistancePair = std::pair<std::make_signed<StorageType>, GridPoint>;
    constexpr auto INF = std::numeric_limits<std::make_signed<StorageType>>::max();

    Eigen::Matrix<StorageType, Rows, Cols, Eigen::RowMajor> distances;
    distances.setConstant(INF);
    distances(start.getX(), start.getY()) = 0;

    std::unordered_map<GridPoint, GridPoint> previous;
    std::priority_queue<DistancePair> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [current_dist, cell] = pq.top();
        pq.pop();

        if (cell == end) {
            // Reconstruct the path
            std::vector<GridPoint> path;
            for (const GridPoint &at = end; at != start; at = previous[at]) {
                path.push_back(at);
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }

        for (const auto& dir : directions) {
            GridPoint neighbor = cell + dir;
            if ((neighbor.getX() >= 0 and neighbor.getX() < Rows) and
                (neighbor.getY() >= 0 and neighbor.getY() < Cols) and
                grid(neighbor.getX(), neighbor.getY()) == simulation::ArenaTile::Empty) {
                auto new_dist = current_dist + 1;
                if (new_dist < distances(neighbor.getX(), neighbor.getY())) {
                    distances(neighbor.getX(), neighbor.getY()) = new_dist;
                    pq.push({new_dist, neighbor});
                    previous[neighbor] = cell;
                }
            }
        }
    }
    throw std::runtime_error("No path found");
}

template<typename StorageType, StorageType Rows, StorageType Cols>
constexpr StorageType dijkstra_constexpr(
    const std::array<std::array<simulation::ArenaTile, Cols>, Rows>& grid,
    simulation::Point<StorageType> current,
    simulation::Point<StorageType> end,
    std::array<std::array<StorageType, Cols>, Rows>& distances,
    std::array<std::array<simulation::Point<StorageType>, Cols>, Rows>& previous,
    const StorageType current_dist = 0) {

    constexpr auto INF = std::numeric_limits<StorageType>::max();

    if (current == end) {
        return current_dist;
    }

    auto currentScore = std::abs(current.getX() - end.getX()) + std::abs(current.getY() - end.getY());

    auto min_dist = INF;
    for (const auto& dir : directions) {
        if (simulation::Point<StorageType> neighbor = current + dir;
            neighbor.getX() >= 0 and neighbor.getX() < Rows and
            neighbor.getY() >= 0 and neighbor.getY() < Cols and
            (grid[neighbor.getY()][neighbor.getX()] == simulation::ArenaTile::Empty or
                grid[neighbor.getY()][neighbor.getX()] == simulation::ArenaTile::Goal)) {
            int new_dist = current_dist + 1;
            if (new_dist < distances[neighbor.getY()][neighbor.getX()]) {
                distances[neighbor.getY()][neighbor.getX()] = new_dist;
                previous[neighbor.getY()][neighbor.getX()] = current;  // Store parent point
                auto newScore = std::abs(neighbor.getX() - end.getX()) + std::abs(neighbor.getY() - end.getY());
                if (newScore < currentScore)
                min_dist = std::min(min_dist, dijkstra_constexpr<StorageType, Rows, Cols>(grid, neighbor, end, distances, previous, new_dist));
            }
        }
    }
    return min_dist;
}

template<typename StorageType, StorageType Rows, StorageType Cols, StorageType MAX_LEN = -1, StorageType LEN = MAX_LEN == -1 ? Rows * Cols : MAX_LEN>
constexpr std::array<simulation::Point<StorageType>, LEN> reconstruct_path(
    const std::array<std::array<simulation::Point<StorageType>, Cols>, Rows>& previous,
    simulation::Point<StorageType> start,
    simulation::Point<StorageType> end) {

    std::array<simulation::Point<StorageType>, LEN> path = create_array<LEN, simulation::Point<StorageType>>({-1, -1});
    size_t index = 0;
    simulation::Point<StorageType> current = end;

    while (current != start and current != simulation::Point<StorageType>{-1, -1}) {
        path[index++] = current;
        current = previous[current.getY()][current.getX()];  // Follow the parent nodes
    }
    // Index points to one past the last element in the array

    // Reverse inplace
    --index; // Now points to last element
    for (auto i = 0; i < index >> 1; ++i) {
        swap(path[i], path[index - i]);
    }

    return path;
}

// Wrapper for easier use of dijkstra_constexpr
template<typename StorageType, StorageType Rows, StorageType Cols, StorageType MAX_LEN = -1, StorageType LEN = MAX_LEN == -1 ? Rows * Cols : MAX_LEN>
constexpr std::array<simulation::Point<StorageType>, LEN> dijkstra_wrapper(
    const std::array<std::array<simulation::ArenaTile, Cols>, Rows>& grid,
    simulation::Point<StorageType> start,
    simulation::Point<StorageType> end) {

    // Create the distance matrix, initialized to INF
    std::array<std::array<StorageType, Cols>, Rows> distances =
        create_array<Rows, std::array<StorageType, Cols>>(
            create_array<Cols, StorageType>(
                std::numeric_limits<StorageType>::max()));
    distances[start.getX()][start.getY()] = 0;

    // Create the previous matrix, initialized to an invalid point
    std::array<std::array<simulation::Point<StorageType>, Cols>, Rows> previous =
        create_array<Rows, std::array<simulation::Point<StorageType>, Cols>>(
            create_array<Cols, simulation::Point<StorageType>>(
                {-1, -1}));

    // Run Dijkstra's algorithm
    auto path = dijkstra_constexpr<StorageType, Rows, Cols>(grid, start, end, distances, previous);

    if (path > MAX_LEN) throw std::runtime_error("Path too large");
    // Reconstruct the path from the previous matrix
    return reconstruct_path<StorageType, Rows, Cols, MAX_LEN>(previous, start, end);
}
*/

#endif //ASTAR_H
