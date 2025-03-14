#ifndef SHARED_POINT_H
#define SHARED_POINT_H

#include "direction.h"
#include "point.h"

namespace shared {
    constexpr auto SHARED_POINT_SIZE = sizeof(simulation::Point<>);

    // Wrapper a shared instance of a point
    template <typename StorageType = simulation::PointStorageType>
    class SharedPoint {
    public:
        SharedPoint(void *buf) : m_pnt(static_cast<simulation::Point<StorageType> *>(buf)) {}
        SharedPoint(void *buf, StorageType x, StorageType y) : SharedPoint(buf) {
            assert(this->m_pnt != nullptr);
            *this->m_pnt = simulation::Point<StorageType>(x, y);
        }
        SharedPoint(void *buf, const simulation::Point<StorageType> &pnt) : SharedPoint(buf, pnt.getX(), pnt.getY()) {}
        SharedPoint(const SharedPoint &) = delete;
        SharedPoint &operator=(const SharedPoint &) = delete;
        SharedPoint(SharedPoint &&other) noexcept : m_pnt(std::move(other.m_pnt)) {}
        SharedPoint &operator=(SharedPoint &&other) noexcept {
            this->m_pnt = std::move(other.m_pnt);
            return *this;
        }

        constexpr simulation::Point<StorageType> operator+(const simulation::Point<StorageType> &other) const { return this->get() + other; }
        constexpr simulation::Point<StorageType> operator+(const SharedPoint &other) const { return this->get() + other.get(); }
        constexpr simulation::Point<StorageType> operator+(const simulation::Direction &other) const { return this->get() + directionToPoint(other); }
        constexpr SharedPoint &operator+=(const simulation::Point<StorageType> &other) { this->get() += other; return *this; }
        constexpr SharedPoint &operator+=(const SharedPoint &other) { this->get() += other.get(); return *this; }
        constexpr SharedPoint &operator+=(const simulation::Direction &other) { this->get() += directionToPoint(other); return *this; }
        constexpr simulation::Point<StorageType> operator-(const simulation::Point<StorageType> &other) const { return this->get() - other; }
        constexpr simulation::Point<StorageType> operator-(const SharedPoint &other) const { return this->get() - other.get(); }
        constexpr SharedPoint &operator-=(const simulation::Point<StorageType> &other) { this->get() -= other; return *this; }
        constexpr SharedPoint &operator-=(const SharedPoint &other) { this->get() -= other.get(); return *this; }
        constexpr bool operator<(const simulation::Point<StorageType> &other) const { return this->get() < other; }
        constexpr bool operator==(const simulation::Point<StorageType> &other) const { return this->get() == other; }
        constexpr bool operator!=(const simulation::Point<StorageType> &other) const { return this->get() != other; }
        constexpr bool operator>(const simulation::Point<StorageType> &other) const { return this->get() > other; }
        constexpr bool operator<(const SharedPoint &other) const { return this->get() < other.get(); }
        constexpr bool operator==(const SharedPoint &other) const { return this->get() == other.get(); }
        constexpr bool operator!=(const SharedPoint &other) const { return this->get() != other.get(); }
        constexpr bool operator>(const SharedPoint &other) const { return this->get() > other.get(); }

        operator std::string() const { return std::string(this->get()); }

        StorageType getX() const { return this->m_pnt->getX(); }
        StorageType getY() const { return this->m_pnt->getY(); }
        void setX(const StorageType &x) { this->m_pnt->setX(x); }
        void setY(const StorageType &y) { this->m_pnt->setY(y); }

        const simulation::Point<StorageType> &get() const { return *this->m_pnt; }
        simulation::Point<StorageType> get() { return *this->m_pnt; }

        [[nodiscard]] SharedPoint copy(void *buf) const {
            assert(buf != nullptr);
            auto *p = static_cast<simulation::Point<StorageType> *>(buf);
            p->setX(this->m_pnt->getX());
            p->setY(this->m_pnt->getY());
            return {buf};
        }
        [[nodiscard]] SharedPoint clone() const {
            return SharedPoint(this->m_pnt);
        }

    private:
        simulation::Point<StorageType> *m_pnt = nullptr;
    };
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const shared::SharedPoint<T> &s) {
    os << s.get();
    return os;
}

#endif //SHARED_POINT_H
