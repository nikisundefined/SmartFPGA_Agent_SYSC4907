#ifndef PLAYER_H
#define PLAYER_H
#include "point.h"
#include <list>

namespace simulation {

    constexpr uint32_t PLAYER_SCORE_PER_GOAL = 100;

    template <typename StorageType = PointStorageType>
    class Player {
    public:
        Player(const Point<StorageType> &p, const uint32_t s, const std::list<Point<StorageType>> &ps) : point(p), score(s), positions(ps) {}
        Player(const Point<StorageType> &p, uint32_t s) : Player(p, s, {}) {}
        Player(const Point<StorageType> &p) : Player(p, 0, {}) {}

        Player copy() const { return Player(point, score, positions); }
        Player clone() const { return this->copy(); }
        void move(const Direction d) {
            if (d == Direction::None) return;
            this->point += d;
        }
        void collectGoal() {
            this->score += PLAYER_SCORE_PER_GOAL;
            this->positions.clear();
            this->positions.push_back(point);
        }

        Point<StorageType> operator+(const Point<StorageType> &other) const { return this->point + other; }
        Point<StorageType> operator+(const Direction &d) const { return this->point + d; }
        Player &operator+=(const Point<StorageType> &other) { this->point += other; return *this; }
        Player &operator+=(const Direction &d) { this->point += d; return *this; }
        Point<StorageType> operator-(const Point<StorageType> &other) const { return this->point - other; }
        bool operator==(const Point<StorageType> &other) const { return this->point == other; }
        operator std::string() const { return std::string(this->point); }
        operator Point<StorageType>() const { return this->point; }

        [[nodiscard]] PointStorageType getX() const { return this->point.getX(); }
        void setX(const PointStorageType other) { this->point.setX(other); }
        [[nodiscard]] PointStorageType getY() const { return this->point.getY(); }
        void setY(const PointStorageType other) { this->point.setY(other); }
        [[nodiscard]] uint32_t getScore() const { return this->score; }
        void setScore(const uint32_t other) { this->score = other; }
        std::list<Point<StorageType>> &getPositions() { return this->positions; }
        void setPositions(const std::list<Point<StorageType>> &other) { this->positions = other; }

#ifdef SIMULATION_JSON
        operator nlohmann::json() const {
            return {
                {"point", nlohmann::json(this->point)},
                {"score", this->score},
                {"positions", nlohmann::json(this->positions)}
            };
        }
#endif

    private:
        Point<StorageType> point;
        uint32_t score;
        std::list<Point<StorageType>> positions;
    };

}

template <typename StorageType>
struct std::hash<simulation::Player<StorageType>> {
    std::size_t operator()(const simulation::Player<StorageType>& s) const noexcept {
        return s.getX() * s.getY() ^ s.getScore();
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const simulation::Player<T> &p) {
    os << std::string(p);
    return os;
}

#endif //PLAYER_H
