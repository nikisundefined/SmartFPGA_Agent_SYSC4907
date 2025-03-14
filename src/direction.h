#ifndef DIRECTION_H
#define DIRECTION_H

#include <eigen3/Eigen/Core>

namespace simulation {
    enum class Direction {
        None = -1,
        Up = 0,
        Right = 1,
        Down = 2,
        Left = 3
    };

    template <typename ValueType = int8_t>
    constexpr Eigen::Vector<ValueType, 4> directionToVector(Direction d) {
        Eigen::Vector<ValueType, 4> vec;
        vec << 0, 0, 0, 0;
        if (d == Direction::None) return vec;
        vec(static_cast<uint8_t>(d)) = 1;
        return vec;
    }

}

#endif //DIRECTION_H
