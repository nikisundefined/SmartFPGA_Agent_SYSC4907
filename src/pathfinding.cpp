#include "pathfinding.h"
#include "arena.h"
/*
void print_grid() {
    for (auto i = 0; i != 23; ++i) {
        for (auto j = 0; j != 23; ++j) {
            switch (simulation::Arena<>::defaultGrid[i][j]) {
                case simulation::ArenaTile::Empty:
                    std::cout << "+";
                break;
                case simulation::ArenaTile::Wall:
                    std::cout << "#";
                break;
                case simulation::ArenaTile::Player:
                    std::cout << "P";
                break;
                case simulation::ArenaTile::Goal:
                    std::cout << "G";
                break;
            }
        }
        std::cout << std::endl;
    }
}

constexpr std::size_t point_count() {
    std::size_t count = 0;
    for (auto i = 0; i != 23; ++i) {
        for (auto j = 0; j != 23; ++j) {
            if (simulation::Arena<>::defaultGrid[i][j] == simulation::ArenaTile::Empty)
                count++;
        }
    }
    return count;
}

template <typename StorageType = simulation::PointStorageType>
constexpr std::array<simulation::Point<StorageType>, point_count()> points() {
    std::array<simulation::Point<StorageType>, point_count()> result = create_array< point_count(), simulation::Point<StorageType>>({-1, -1});
    std::size_t index = 0;
    for (StorageType i = 0; i != 23; ++i) {
        for (StorageType j = 0; j != 23; ++j) {
            if (simulation::Arena<>::defaultGrid[i][j] == simulation::ArenaTile::Empty)
                result[index] = {i, j};
        }
    }
    return result;
}

template <std::size_t MAX_LEN, typename StorageType = simulation::PointStorageType>
constexpr std::array<simulation::Point<StorageType>, MAX_LEN> get_path(const simulation::Point<StorageType> &start, const simulation::Point<StorageType> &end) {
    return dijkstra_wrapper<StorageType, 23, 23, MAX_LEN>(simulation::Arena<>::defaultGrid, start, end);
}

constexpr auto get_paths() {
    constexpr auto pc = point_count();
    auto ps = points();
    std::array<std::array<simulation::Point<>, simulation::Arena<>::pathMaxLen>, pc * pc> paths = create_array<pc * pc, std::array<simulation::Point<>, simulation::Arena<>::pathMaxLen>>(
                create_array<simulation::Arena<>::pathMaxLen, simulation::Point<>>({-1, -1}));
    std::size_t index = 0;
    for (const auto &start : ps) {
        for (const auto &end : ps) {
            auto tmp = get_path<simulation::Arena<>::pathMaxLen>(start, end);
            std::size_t i = 0;
            for (const auto &path : tmp) {
                paths[index][i++] = path;
            }
            index++;
        }
    }
    return paths;
}
*/

#include <iostream>

int main() {
    try {
        const simulation::Arena<> arena;
        const auto path = arena.distance();
        const auto count = path.size();
        std::cout << "Path Length: " << count << std::endl;
        for (const auto &elem : path) {
            std::cout << "(" << +elem.getX() << ", " << +elem.getY() << ")" << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
