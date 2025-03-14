#ifndef POINT_H
#define POINT_H

#include <eigen3/Eigen/Core>
#include <cmath>
#include <string>
#include <sstream>
#ifdef SIMULATION_JSON
#include <nlohmann/json.hpp>
#endif
#include "direction.h"

namespace simulation {

    using PointStorageType = int8_t;

    template <typename StorageType = PointStorageType>
    class Point {
    public:
        constexpr Point(const Point &p) : x(p.x), y(p.y) {}
        constexpr Point(const Point &&p) noexcept : x(p.x), y(p.y) {}
        constexpr Point(Eigen::Ref<const Eigen::Vector<StorageType, 4>> vec) : x(vec.y() - vec.w()), y(vec.x() - vec.z()) {}
        constexpr Point(StorageType x, StorageType y) : x(x), y(y) {}

        constexpr Point &operator=(const Point &) = default;
        constexpr Point &operator=(Point &&) noexcept = default;


        [[nodiscard]] constexpr StorageType getX() const { return this->x; }
        constexpr void setX(StorageType x) { this->x = x; }
        [[nodiscard]] constexpr StorageType getY() const { return this->y; }
        constexpr void setY(StorageType y) { this->y = y; }

        template <typename S = StorageType>
        constexpr Eigen::Vector<S, 4> magnitude() const {
            Eigen::Vector<S, 4> magnitude;
            magnitude << -this->y, this->x, this->y, -this->x;
            return magnitude;
        }
        [[nodiscard]] constexpr Direction direction() const {
            assert(this->x == -1 or this->x == 0 or this->x == 1 && "Point must contain a x coordinate that is either -1,0,1");
            assert(this->y == -1 or this->y == 0 or this->y == 1 && "Point must contain a y coordinate that is either -1,0,1");
            if (this->x == 0 and this->y == 0) return Direction::None;

            if (this->x == 0) {
                return this->y > 0 ? Direction::Up : Direction::Down;
            }
            if (this->y == 0) {
                return this->x > 0 ? Direction::Right : Direction::Left;
            }
            throw std::runtime_error("Direction must be zero in either x or y direction");
        }
        [[nodiscard]] constexpr double distance(const Point &other) const { return std::sqrt(std::pow(this->x - other.x, 2) + std::pow(this->y - other.y, 2)); }
        [[nodiscard]] constexpr Point copy() const { return Point(this->x, this->y); }
        [[nodiscard]] constexpr Point clone() const { return this->copy(); }

        constexpr Point operator+(const Point &other) const { return Point(this->x + other.x, this->y + other.y); }
        constexpr Point operator+(const Direction &other) const;
        constexpr Point &operator+=(const Point &other) { this->x += other.x; this->y += other.y; return *this; }
        constexpr Point &operator+=(const Direction &other);
        constexpr Point operator-(const Point &other) const { return Point(this->x - other.x, this->y - other.y); }
        constexpr Point &operator-=(const Point &other) { this->x -= other.x; this->y -= other.y; return *this; }
        constexpr bool operator<(const Point &other) const { return this->x < other.x or this->y < other.y; }
        constexpr bool operator==(const Point &other) const { return this->x == other.x and this->y == other.y; }
        constexpr bool operator!=(const Point &other) const { return this->x != other.x or this->y != other.y; }
        constexpr bool operator>(const Point &other) const { return this->x > other.x or this->y > other.y; }

        operator std::string() const {
            std::stringstream ss;
            ss << "(" << static_cast<int32_t>(this->x) << ", " << static_cast<int32_t>(this->y) << ")";
            return ss.str();
        }
#ifdef SIMULATION_JSON
        operator nlohmann::json() const {
            return {{"x", static_cast<int32_t>(this->x)}, {"y", static_cast<int32_t>(this->y)}};
        }
#endif
    private:
        constexpr friend void swap(Point &lhs, Point &rhs) noexcept {
            auto tmp_x = lhs.x;
            auto tmp_y = lhs.y;
            lhs.x = rhs.x;
            lhs.y = rhs.y;
            rhs.x = tmp_x;
            rhs.y = tmp_y;
        }

        StorageType x;
        StorageType y;
    };

    constexpr Point<> directionToPoint(const Direction d) {
        switch (d) {
            case Direction::None:
                return {0, 0};
            case Direction::Up:
                return {0, -1};
            case Direction::Right:
                return {1, 0};
            case Direction::Down:
                return {0, 1};
            case Direction::Left:
                return {-1, 0};
        }
        throw std::invalid_argument("Invalid direction");
    }

    template <typename StorageType>
    constexpr Point<StorageType> Point<StorageType>::operator+(const Direction &other) const {
        return *this + directionToPoint(other);
    }

    template <typename StorageType>
    constexpr Point<StorageType> &Point<StorageType>::operator+=(const Direction &other) {
        return *this += directionToPoint(other);
    }

} // simulation

template<typename StorageType>
struct std::hash<simulation::Point<StorageType>> {
std::size_t operator()(const simulation::Point<StorageType>& s) const noexcept {
    return s.getX() * s.getY();
}
};

template <typename T>
 std::ostream &operator<<(std::ostream &os, const simulation::Point<T> &p) {
    os << std::string(p);
    return os;
}

#endif //POINT_H
