#ifndef SHARED_PATH_CACHE_H
#define SHARED_PATH_CACHE_H

#include "path_cache.h"

#include <numeric>

namespace shared {

    constexpr auto SHARED_PATH_CACHE_SIZE = -1;

    template <typename StorageType = simulation::PointStorageType>
    class SharedPathCache {
    public:
        SharedPathCache(simulation::Point<StorageType> *buf) : m_size(-1), cache(buf) {
            assert(buf != nullptr);
        }
        SharedPathCache(simulation::Point<StorageType> *buf, std::pair<simulation::PathPair,
            std::pair<std::size_t, std::size_t>> keys[], const std::size_t keys_size) : m_size(-1), cache(buf) {
            assert(buf != nullptr);
            assert(keys != nullptr);
            assert(keys_size > 0);

            this->key.reserve(keys_size);
            for (auto i = 0; i < keys_size; i++) {
                this->key.emplace(keys[i]);
            }
        }
        SharedPathCache(simulation::Point<StorageType> *buf, const simulation::PathCache &c) : m_size(c.size()), cache(buf) {
            this->key.reserve(c.size());
            std::size_t offset = 0;
            for (const auto &[key, value] : c.getCache()) {
                this->key.emplace(key, std::pair{offset, value.size()});
                offset += value.size() * sizeof(simulation::Point<StorageType>);
            }
        }

        auto keys() const { return this->key; }
        void loadKeys(const std::vector<std::pair<simulation::PathPair, std::size_t>> &k) {
            std::size_t offset = 0;
            for (const auto &[key, value] : k) {
                this->key.emplace(key, std::pair{offset, value});
                offset += value;
            }
            this->m_size = k.size();
        }

        std::vector<simulation::Point<StorageType>> operator[](const simulation::PathPair &k) const {
            assert(this->key.find(k) != this->key.end());
            const auto &[offset, size] = this->key.at(k);
            std::vector<simulation::Point<StorageType>> output(&this->cache[offset], &this->cache[offset + size]);
            return output;
        }
        bool contains(const simulation::PathPair &k) const {
            assert(this->m_size != -1 and this->m_size != 0);
            return this->key.find(k) != this->key.end();
        }
        std::size_t count() const {
            return std::accumulate(this->key.begin(), this->key.end(), 0,
                [](auto &s, const auto &t) { return s + t.second.second; });
        }
        std::size_t size() const { return this->m_size; }
#ifdef SHARED_JSON


#endif
    private:
        std::size_t m_size;
        simulation::Point<StorageType> *cache; // The actual points stored in the cache
        std::unordered_map<simulation::PathPair, std::pair<std::size_t, std::size_t>> key; // Maps (start, end) to (offset, len)
    };

}

#endif //SHARED_PATH_CACHE_H
