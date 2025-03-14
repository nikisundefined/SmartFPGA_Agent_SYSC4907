#ifndef PATH_PAIR_H
#define PATH_PAIR_H

#include "point.h"
#include <string_view>

namespace simulation {

    class PathPair {
    public:
        template <typename StorageType = PointStorageType>
        PathPair(std::string_view str): start(0, 0), end(0, 0) {
            str = str.substr(1, str.size() - 2);

            auto start = str.find('(');
            auto end = str.find(')', start);
            auto mid = str.find(',', start);
            auto x = std::atoi(str.substr(start + 1, mid - start - 1).data());
            auto y = std::atoi(str.substr(mid + 1, end - mid - 1).data());
            Point<StorageType> p1(x, y);

            start = str.find('(', end);
            end = str.find(')', start);
            mid = str.find(',', start);
            x = std::atoi(str.substr(start + 1, mid - start - 1).data());
            y = std::atoi(str.substr(mid + 1, end - mid - 1).data());
            Point<StorageType> p2(x, y);

            this->start = p1;
            this->end = p2;
        }

        constexpr PathPair(const Point<> &start, const Point<> &end) : start(start), end(end) {}

        Point<> getStart() const noexcept { return this->start; }
        Point<> getEnd() const noexcept { return this->end; }

        constexpr bool operator==(const PathPair &other) const { return this->start == other.start && this->end == other.end; }
        operator std::string() const {
            std::stringstream ss;
            ss << "[" << std::string(this->start) << ", " << std::string(this->end) << "]";
            return ss.str();
        }
#ifdef SIMULATION_JSON
        operator nlohmann::json() const {
            return {
                {"start", this->start},
                {"end", this->end}
            };
        }
#endif

        std::ostream &operator<<(std::ostream &os) const;

    private:
        Point<> start, end;
    };

}

template <>
struct std::hash<simulation::PathPair> {
    std::size_t operator()(const simulation::PathPair& s) const noexcept {
        return std::hash<simulation::Point<>>()(s.getStart()) ^ std::hash<simulation::Point<>>()(s.getEnd());
    }
};

inline std::ostream &operator<<(std::ostream &os, const simulation::PathPair &p) {
    os << std::string(p);
    return os;
}

#endif //PATH_PAIR_H
