#ifndef SHARED_PLAYER_H
#define SHARED_PLAYER_H

#include "player.h"
#include "shared_point.h"

namespace shared {

    constexpr auto SHARED_PLAYER_SIZE = SHARED_POINT_SIZE + sizeof(std::declval<simulation::Player<>>().getScore());

    template <typename StorageType = simulation::PointStorageType>
    class SharedPlayer {
    public:
        SharedPlayer(void *p) : m_player(static_cast<simulation::Player<StorageType> *>(p)) {
            assert(p != nullptr);
        }
        SharedPlayer(void *buf, const simulation::Point<StorageType> &p) : SharedPlayer(buf) {
            *this->m_player = p;
        }
        SharedPlayer(void *buf, const SharedPoint<StorageType> &sp) : SharedPlayer(buf, sp.get()) {}


        SharedPlayer copy(void *buf) const {
            assert(buf != nullptr);
            auto *p = static_cast<simulation::Player<StorageType> *>(buf);
            p->setX(this->m_player->getX());
            p->setY(this->m_player->getY());
            p->setScore(this->m_player->getScore());
            p->setPositions(this->m_player->getPositions());
            return SharedPlayer(p);
        }
        SharedPlayer clone() const {
            return SharedPlayer(this->m_player);
        }

        simulation::Point<StorageType> operator+(const simulation::Point<StorageType> &other) const { return *this->m_player + other; }
        simulation::Point<StorageType> operator+(const simulation::Direction &d) const { return *this->m_player + d; }
        SharedPlayer &operator+=(const simulation::Point<StorageType> &other) { *this->m_player += other; return *this; }
        SharedPlayer &operator+=(const simulation::Direction &d) { *this->m_player += d; return *this; }
        simulation::Point<StorageType> operator-(const simulation::Point<StorageType> &other) const { return *this->m_player - other; }
        bool operator==(const simulation::Point<StorageType> &other) const { return *this->m_player == other; }
        operator std::string() const { return std::string(*this->m_player); }
        operator simulation::Point<StorageType>() const { return simulation::Point<StorageType>(*this->m_player); }

        const simulation::Player<StorageType> &getPlayer() const { return *this->m_player; }
        simulation::Player<StorageType> &getPlayer() { return *this->m_player;}

        [[nodiscard]] StorageType getX() const { return this->m_player->getX(); }
        void setX(const StorageType other) { this->m_player->setX(other); }
        [[nodiscard]] StorageType getY() const { return this->m_player->getY(); }
        void setY(const StorageType other) { this->m_player->setY(other); }
        [[nodiscard]] uint32_t getScore() const { return this->m_player->getScore(); }
        void setScore(const uint32_t other) { this->m_player->setScore(other); }
        std::list<simulation::Point<StorageType>> &getPositions() { return this->m_player->getPositions(); }
        void setPositions(const std::list<simulation::Point<StorageType>> &other) { this->m_player->setPositions(other); }
    private:
        simulation::Player<StorageType> *m_player;
    };

}

template <typename StorageType>
struct std::hash<shared::SharedPlayer<StorageType>> {
    std::size_t operator()(const shared::SharedPlayer<StorageType>& s) const noexcept {
        return std::hash<simulation::Player<StorageType>>()(s.getPlayer());
    }
};

#endif //SHARED_PLAYER_H
