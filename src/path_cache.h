#ifndef PATH_CACHE_H
#define PATH_CACHE_H

#include "path_pair.h"
#include <unordered_map>
#include <list>
#include <filesystem>
#include <fstream>
#include <memory>

namespace simulation {

    class PathCache {
    public:
        PathCache() = default;
#ifdef SIMULATION_JSON
        PathCache(std::string_view json_string) {
            this->loads(json_string);
        }
        PathCache(const std::filesystem::path& file) {
            if (not is_regular_file(file)) throw std::invalid_argument("File is not a regular file or does not exist");
            std::string str;
            {
                std::ifstream in(file);
                in >> str;
            }
            this->loads(std::string_view(str));
        }

        void loads(std::string_view json_string) {
            nlohmann::json json = nlohmann::json::parse(json_string);
            this->cache.reserve(json.size());
            for (const auto &[k, v] : json.items()) {
                const auto key = PathPair(k);
                const auto &m = v.get<std::vector<std::string>>();
                std::list value(m.size(), Point<>(0, 0));
                std::transform(m.begin(), m.end(), value.begin(), [](const nlohmann::json &e) {
                    return Point(e.at("x").get<PointStorageType>(), e.at("y").get<PointStorageType>());
                });
                this->cache.emplace(key, std::move(value));
            }
        }

        operator nlohmann::json() const {
            return {this->cache};
        }
#endif

        size_t count() const {
            size_t count = 0;
            for (const auto &[_, v] : this->cache) {
                count += v.size();
            }
            return count;
        }
        bool contains(const PathPair &key) const { return cache.find(key) != cache.end(); }
        std::size_t size() const { return cache.size(); }

        auto operator[](const PathPair &k) const { return cache.at(k); }

        const auto &getCache() const { return cache; }
    private:
        std::unordered_map<PathPair, std::list<Point<>>> cache;
    };

#ifdef SIMULATION_FROZEN
#include <frozen/unordered_map.h>

    template <std::size_t N, std::size_t L, typename StorageType = PointStorageType>
    class FrozenPathCache : public PathCache {
    public:
        constexpr FrozenPathCache(std::array<std::pair<PathPair, std::array<Point<StorageType>, L>>, N> &c) : cache() {
            for (const auto &[k, v] : c) {
                this->cache.emplace(k, v);
            }
        }

    private:
        const frozen::unordered_map<PathPair, std::array<Point<StorageType>, L>, N> cache;
    };
#endif
}

#endif //PATH_CACHE_H
