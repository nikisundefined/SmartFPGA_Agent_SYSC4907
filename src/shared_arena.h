#ifndef SHARED_ARENA_H
#define SHARED_ARENA_H

#include "arena.h"

namespace shared {

    constexpr auto SHARED_ARENA_SIZE = sizeof(simulation::Arena<>);

    class SharedArena {
    public:
        SharedArena(void *buf) : m_arena(static_cast<simulation::Arena<> *>(buf)) {
            *this->m_arena = simulation::Arena();
        }
        SharedArena(void *buf, const simulation::Arena<> &arena) : m_arena(static_cast<simulation::Arena<> *>(buf)) {
            *this->m_arena = arena;
        }

        operator std::string() const { return std::string(*this->m_arena); }

        [[nodiscard]] const simulation::Arena<> &get() const { return *m_arena; }
        [[nodiscard]] simulation::Arena<> &get() { return *m_arena; }
    private:
        simulation::Arena<> *m_arena;
    };

}

#endif //SHARED_ARENA_H
