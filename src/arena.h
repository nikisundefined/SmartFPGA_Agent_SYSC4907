#ifndef ARENA_H
#define ARENA_H

#include <mutex>
#include <cmath>
#include <random>

#ifdef SMART_AGENT_JSON
#include <nlohmann/json.hpp>
#endif

#include "point.h"
#include "player.h"
#include "path_cache.h"
#include "pathfinding.h"

namespace simulation {

    template <std::size_t N = 23, std::size_t M = 23, typename StorageType = PointStorageType>
    class Arena {
    public:
    	using GridPoint = Point<StorageType>;
    	static PathCache pathCache;
    	static std::recursive_mutex pathCacheLock;
    	constexpr static GridPoint playerStart = {3, 3};
    	constexpr static GridPoint goalStart = {11, 9};
    	constexpr static std::size_t pathMaxLen = 44;
    	constexpr static std::array<std::array<ArenaTile, 23>, 23> defaultGrid = {{
    		{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Player, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Goal, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Empty, ArenaTile::Wall},
			{ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall, ArenaTile::Wall},
			}};

    	const std::size_t n = N;
    	const std::size_t m = M;

    	Arena() : Arena(defaultGrid, playerStart, goalStart) {}
    	Arena(const std::array<std::array<ArenaTile, M>, N> &grid, const GridPoint &playerStart, const GridPoint &goalStart) :
		finder(grid), player(playerStart), goal(goalStart) {
    		// Copy the grid into the local arena object
    		this->grid = pathfinding::create_matrix<N, M, ArenaTile>(ArenaTile::Empty);
    		for (auto i = 0; i != N; ++i) {
    			for (auto j = 0; j != M; ++j) {
    				this->grid[j][i] = grid[j][i];
    			}
    		}
    		// Ensure the starting locations are valid for this grid
    		assert(this->tile(this->player, 0, 0) == ArenaTile::Player);
    		assert(this->tile(this->goal, 0, 0) == ArenaTile::Goal);
    	}
    	Arena(const Arena &a) : Arena(a.grid, a.player, a.goal) {}
    	Arena &operator=(const Arena &a) {
    		// Template asserts that this arena is the same size as the other arena
    		for (auto i = 0; i != N; ++i) {
    			for (auto j = 0; j != M; ++j) {
    				this->grid[i][j] = a.grid[i][j];
    			}
    		}
			this->player = a.player;
    		this->goal = a.goal;
    		this->finder = a.finder;
    		return *this;
    	}

    	[[nodiscard]] bool onGoal() const { return player == goal; }
    	[[nodiscard]] Direction bestDirection(const GridPoint &start, const GridPoint &end) const {
    		auto distance = this->distance(start, end);
    		if (distance.size() == 0) return Direction::None;
    		auto delta = distance.front() - start;
    		return delta.direction();
    	}

    	void move(Direction d) {
    		// Ensure that if the player was on the goal and moved off it, it remains a goal
    		tile(this->player) = onGoal() ? ArenaTile::Goal : ArenaTile::Empty;

    		switch (d) {
    			case Direction::None:
    				tile(this->player) = ArenaTile::Player;
    			throw std::invalid_argument("bad direction");
    			case Direction::Up:
    				if (this->player.getY() == 0 or tile(this->player, 0, -1) != ArenaTile::Wall)
    					this->player += d;
    			break;
    			case Direction::Right:
    				if (this->player.getX() == N - 1 or tile(this->player, 1, 0) != ArenaTile::Wall)
    					this->player += d;
    			break;
    			case Direction::Down:
    				if (this->player.getY() == M - 1 or tile(this->player, 0, 1) != ArenaTile::Wall)
    					this->player += d;
    			break;
    			case Direction::Left:
    				if (this->player.getX() == 0 or tile(this->player, -1, 0) != ArenaTile::Wall)
    					this->player += d;
    			break;
    		}

    		if (this->player.getX() >= N) this->player.setX(0);
    		else if (this->player.getX() < 0) this->player.setX(N - 1);
    		if (this->player.getY() >= M) this->player.setY(0);
    		else if (this->player.getY() < 0) this->player.setY(M - 1);

    		tile(this->player) = ArenaTile::Player;
    	}
    	[[nodiscard]] Eigen::Vector<StorageType, 4> detection() const {
    		Eigen::Vector<StorageType, 4> result;
    		result.setZero();

    		constexpr auto Up = static_cast<int>(Direction::Up);
    		constexpr auto Down = static_cast<int>(Direction::Down);
    		constexpr auto Left = static_cast<int>(Direction::Left);
    		constexpr auto Right = static_cast<int>(Direction::Right);

    		// Scan upwards
    		while (this->tile(this->player, 0, -result[Up]) == ArenaTile::Empty)
    			result[Up] += 1;

    		// Scan downwards
    		while (this->tile(this->player, 0, result[Down]) == ArenaTile::Empty)
    			result[Down] += 1;

    		while (this->tile(this->player, -result[Left], 0) == ArenaTile::Empty)
    			result[Left] += 1;

    		while (this->tile(this->player, result[Right], 0) == ArenaTile::Empty)
    			result[Right] += 1;

    		return result;
    	}
    	[[nodiscard]] constexpr double absoluteDistance() const { return this->goal.distance(this->player); }
    	[[nodiscard]] std::list<GridPoint> distance(const GridPoint &start, const GridPoint &end) const {
    		if (this->tile(start) == ArenaTile::Wall or this->tile(end) == ArenaTile::Wall) throw std::invalid_argument("Start and End must be empty tiles in this grid");
    		if (not inBounds(start) or not inBounds(end)) throw std::invalid_argument("Start and End must within the bounds of the grid");
    		return finder.path(start, end);
    	}
    	[[nodiscard]] std::list<GridPoint> distance(const GridPoint &start) const { return this->distance(start, this->goal); }
    	[[nodiscard]] std::list<GridPoint> distance() const { return finder.path(this->player, this->goal); }

    	[[nodiscard]] constexpr Player<StorageType> getPlayer() const { return player; }
    	void setPlayer(const GridPoint &p) { this->player.setX(p.getX()); this->player.setY(p.getY()); }
    	[[nodiscard]] GridPoint getGoal() const { return goal; }
    	void setGoal() {
    		if (onGoal()) {
    			tile(this->goal) = ArenaTile::Player;
    			this->player.collectGoal();
    		} else {
    			tile(this->goal) = ArenaTile::Empty;
    		}
    		this->goal = randomPoint();
    		tile(this->goal) = ArenaTile::Goal;
    	}
    	const std::array<std::array<ArenaTile, M>, N> &getGrid() const { return this->grid; }

    	operator std::string() const {
    		std::ostringstream ss;
    		for (auto i = 0; i != N; ++i) {
    			for (auto j = 0; j != M; ++j) {
    				switch (tile(i, j)) {
    					case ArenaTile::Empty:
    						ss << " "; break;
    					case ArenaTile::Wall:
    						ss << "#"; break;
    					case ArenaTile::Player:
    						ss << "P"; break;
    					case ArenaTile::Goal:
    						ss << "G"; break;
    					default:
    						throw std::runtime_error("Unknown tile type");
    				}
    			}
    		}
    		return ss.str();
    	}
#ifdef SMART_AGENT_JSON
    	operator nlohmann::json() const {
    		nlohmann::json j;
    		j["player"] = this->player;
    		j["goal"] = this->goal;
    		return j;
    	}
#endif
    private:
    	GridPoint randomPoint() {
    		// Create random device that is persisted between all function calls
    		static std::random_device rd;
    		static std::mt19937 gen(rd());
    		static std::uniform_int_distribution<StorageType> dist(0, std::max(N, M));

    		auto randX = dist(gen) % N;
    		auto randY = dist(gen) % M;

    		// Keep generating random points until the selected point is not a Wall
    		while (tile(randX, randY) == ArenaTile::Wall) {
    			randX = dist(gen) % N;
    			randY = dist(gen) % M;
    		}

    		return {static_cast<StorageType>(randX), static_cast<StorageType>(randY)};
    	}
    	[[nodiscard]] constexpr ArenaTile &tile(const StorageType &x, const StorageType &y) { return this->grid[y][x]; }
    	[[nodiscard]] constexpr ArenaTile &tile(const GridPoint &p, const StorageType offsetX = 0, const StorageType offsetY = 0) { return this->grid[p.getY() + offsetY][p.getX() + offsetX]; }
    	[[nodiscard]] constexpr const ArenaTile &tile(const StorageType &x, const StorageType &y) const { return this->grid[y][x]; }
    	[[nodiscard]] constexpr const ArenaTile &tile(const GridPoint &p, const StorageType &offsetX = 0, const StorageType &offsetY = 0) const { return this->grid[p.getY() + offsetY][p.getX() + offsetX]; }
    	[[nodiscard]] constexpr static bool inBounds(const GridPoint &p) { return p.getX() > 0 and p.getX() < N and p.getY() > 0 and p.getY() < M; }

        std::array<std::array<ArenaTile, M>, N> grid;
    	mutable pathfinding::Dijikstra2D<N, M, StorageType> finder;
        Player<StorageType> player;
        GridPoint goal;
    };
} // simulation

template <std::size_t N, std::size_t M, typename T>
std::ostream &operator<<(std::ostream &os, const simulation::Arena<N, M, T> &a) {
	os << std::string(a);
	return os;
}

#endif //ARENA_H
