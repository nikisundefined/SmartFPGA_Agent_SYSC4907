#!/usr/bin/env python3

import multiprocessing.synchronize
import numpy as np
import math
import astar
import logging
from typing import Union
from enum import IntEnum

log: logging.Logger = logging.getLogger('simulation')
log.setLevel(logging.DEBUG)

class Point:

    # Convert a 4D numpy array into a point represented as: [-Y, X, Y, -X]
    @classmethod
    def fromvector(cls, vec: np.ndarray) -> 'Point':
        assert vec.shape == (4,), f"Invalid shape for vector: {vec.shape}"
        return cls(vec[1] - vec[3], vec[2] - vec[0])

    # Convert a Point object into a 4D vector of magnitudes represented as [-Y, X, Y, -X]
    def asmagnitude(self, dtype: np.dtype = np.float64) -> np.ndarray:
        tmp: np.ndarray = np.array([-self.y, self.x, self.y, -self.x], dtype=dtype)
        return np.maximum(tmp, np.zeros(tmp.shape))
    
    # Convert a Point object into a direction based on the 
    def asdirection(self) -> 'Direction':
        if self.x > 1 or self.y > 1 or self.x < -1 or self.y < -1:
            raise ValueError(f"Point must contain only a single direction-like value: {self}")
        if self.x and self.y:
            raise ValueError(f"Point should only contain a value in one direction (x or y): {self}")
        
        if self.x:
            return Direction.LEFT if self.x < 0 else Direction.RIGHT
        return Direction.UP if self.y < 0 else Direction.DOWN
    
    # Get the absolute distance to another point
    def distance(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    # Return a copy of the current object
    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
    
    # Add 'Point'-like objects to this point object
    def __add__(self, other: Union['Point', 'Direction', 'Player']) -> 'Point':
        if other is Direction:
            return self + other.topoint()
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    # Is this point less than another, defined as: p.x < o.x and p.y < o.y
    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x or self.y < other.y
    
    def __gt__(self, other: 'Point') -> bool:
        return self.x > other.x or self.y > other.y
    
    def __eq__(self, value: 'Point') -> bool:
        return self.x == value.x and self.y == value.y
    
    def __hash__(self):
        return hash(str(self))

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    def __json__(self) -> dict[str]:
        return {
            'x': int(self.x),
            'y': int(self.y)
        }
        
# An enumeration representing a direction on the grid
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # Convert this direction into a 4D numpy vector
    def tovector(self) -> np.ndarray:
        match self:
            case self.UP:
                return np.array([1, 0, 0, 0])
            case self.RIGHT:
                return np.array([0, 1, 0, 0])
            case self.DOWN:
                return np.array([0, 0, 1, 0])
            case self.LEFT:
                return np.array([0, 0, 0, 1])
    
    # Convert this direction into a point 
    def topoint(self) -> Point:
        return Point.fromvector(self.tovector())

# Holds all information about the player
class Player:
    # Convert a point into a player
    @classmethod
    def frompoint(cls, p: Point) -> 'Player':
        return cls(p.x, p.y, 0, [])
    
    # Convert a set of coordinates into a player
    @classmethod
    def fromcoordinate(cls, x: int, y: int) -> 'Player':
        return cls(x, y, 0, [])

    def __init__(self, x: int, y: int, score: int, positions: list[Point]):
        self.point: Point = Point(x, y)
        self.score: int = score
        self.positions: list[Point] = positions

    # Return a copy of this player object
    def copy(self) -> 'Player':
        return Player(self.point.x, self.point.y, self.score, self.positions.copy())
    
    # Move the player in the specified direction
    def move(self, dir: Direction):
        match int(dir):
            case int(Direction.UP):
                self.y -= 1
            case int(Direction.DOWN):
                self.y += 1
            case int(Direction.LEFT):
                self.x -= 1
            case int(Direction.RIGHT):
                self.x += 1
    
    # Adds a point to this player object (helper method to make math easier)
    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    # Increments the players score for when the player collects a goal
    def collect_goal(self):
        self.score += 100
    
    def __str__(self) -> str:
        return str(self.point)

    def __json__(self) -> dict[str]:
        return {
            'point': self.point.__json__(),
            'score': self.score,
            'positions': [x.__json__() for x in self.positions]
        }

    def __hash__(self):
        return hash(str(self.__json__()))

    # Attribute access to the x coordinate of the player
    @property
    def x(self) -> int:
        return self.point.x
    
    @x.setter
    def x(self, x: int) -> None:
        self.point.x = x
    
    # Attribute access to the y coordinate of the player
    @property
    def y(self) -> int:
        return self.point.y
    
    @y.setter
    def y(self, y: int) -> None:
        self.point.y = y

# Helper class for caching A* paths
class PathPair:
    @classmethod
    def fromstr(cls, s: str) -> 'PathPair':
        s = s.strip('[]')
        nums = s.split(',')

        p = []
        for num in nums:
            num = num.strip(' ()')
            p.append(int(num))
        return cls(Point(p[0], p[1]), Point(p[2], p[3]))

    def __init__(self, start: Point, end: Point):
        self.start: Point = start.copy()
        self.end: Point = end.copy()

    def __eq__(self, other: 'PathPair') -> bool:
        return self.start == other.start and self.end == other.end
    
    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"
    
    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

# The arena containing the player
class Arena:
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    GOAL = 3

    _player_start: Point = Point(3,3) # Initial starting point for the player
    _goal_start: Point = Point(11, 9) # Initial starting point for the goal
    
    def __init__(self, n: int = 23, m: int = 23):
        self.n: int = n
        self.m: int = m
        self.player: Player = Player.frompoint(Arena._player_start)
        self.goal: Point = Point(Arena._goal_start.x, Arena._goal_start.y)
        self.grid: np.ndarray = Arena._create_grid()
        self.path: dict[PathPair, list[Point]] = {}
        assert self.grid[self.goal.y][self.goal.x] != Arena.WALL
    
    def __json__(self) -> dict[str]:
        return {
            'n': self.n,
            'm': self.m,
            'player': self.player.__json__(),
            'goal': self.goal.__json__(),
            'grid': self.grid.tolist(),
        }
    
    @classmethod
    def _create_grid(cls) -> np.ndarray:
        # Generate the arena as a numpy array that is mirrored horizontally then vertically
        n_quad = 11
        grid_quad = np.array([
            [cls.WALL]*n_quad,
            [cls.WALL, cls.EMPTY, cls.EMPTY, cls.EMPTY, cls.WALL] + [cls.EMPTY] * 6,
            [cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY] + [cls.WALL] * 5,
            [cls.WALL] + [cls.EMPTY] * 7 + [cls.WALL, cls.EMPTY, cls.EMPTY],
            [cls.WALL, cls.WALL, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL],
            [cls.WALL, cls.EMPTY, cls.EMPTY, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL],
            [cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL],
            [cls.WALL, cls.EMPTY, cls.EMPTY, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.EMPTY, cls.EMPTY, cls.EMPTY],
            [cls.WALL] * 5 + [cls.EMPTY, cls.WALL, cls.EMPTY] + [cls.WALL] * 3,
            [cls.WALL] + [cls.EMPTY] * 5 + [cls.WALL] + [cls.EMPTY] * 4,
            [cls.WALL, cls.EMPTY, cls.WALL, cls.WALL, cls.WALL, cls.WALL, cls.WALL, cls.EMPTY, cls.WALL, cls.WALL, cls.WALL]
        ], np.float64)
        assert grid_quad.shape == (11, 11), f"Invalid grid shape: {grid_quad.shape}"
        grid_horizontal_spacer = np.array([[cls.EMPTY] * 8 + [cls.WALL] + [cls.EMPTY] * 5 + [cls.WALL] + [cls.EMPTY] * 8], np.float64) # The line between the bottom and top halves
        grid_vertical_spacer = np.array([[cls.WALL, cls.EMPTY, cls.EMPTY, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.WALL, cls.EMPTY, cls.EMPTY]], np.float64).transpose() # The line between the left and right halves
        grid = np.concatenate([grid_quad, grid_vertical_spacer], axis=1) # Append the vertical spacer to the quadrant
        grid = np.concatenate([grid, np.flip(grid_quad, axis=1)], axis=1) # Append the flipped quadrant
        grid_upper = grid.copy() # Make a copy of the upper half of the grid
        grid = np.concatenate([grid, grid_horizontal_spacer], axis=0) # Append the horizontal spacer to the upper half
        grid = np.concatenate([grid, np.flip(grid_upper, axis=0)], axis=0) # Append the flipped upper half
        grid[12][11] = cls.WALL # The only asymmetical part of the map
        grid[Arena._player_start.y][Arena._player_start.x] = cls.PLAYER # Default player position
        grid[Arena._goal_start.y][Arena._goal_start.x] = cls.GOAL # Default goal position
        return grid

    # Polymorphic method to access _tile_pnt and _tile_pos
    def _tile(self, *args, **kwargs) -> int:
        if type(args[0]) == int:
            return self._tile_pos(args[0], args[1])
        elif 'x' in kwargs and 'y' in kwargs:
            return self._tile_pos(kwargs['x'], kwargs['y'])
        elif type(args[0]) == Point or type(args[0]) == Player:
            return self._tile_pnt(args[0], kwargs.get('offset_x', 0), kwargs.get('offset_y', 0))
        elif 'p' in kwargs:
            return self._tile_pnt(kwargs['p'], kwargs.get('offset_x', 0), kwargs.get('offset_y', 0))

    # Get the tile at the point or players location offset by the given amount
    def _tile_pnt(self, p: Point | Player, offset_x: int = 0, offset_y: int = 0) -> int:
        return self.grid[p.y + offset_y][p.x + offset_x]
    
    # Get the tile at the given location
    def _tile_pos(self, x: int, y: int) -> int:
        return self.grid[y][x]
    
    # Check if the player is on the goal
    def on_goal(self) -> bool:
        return self.player == self.goal
    
    # Returns the best direction to go if starting at start and going to end
    def best_direction(self, start: Point | None = None, end: Point | None = None) -> Direction:
        # Set defaults for start and end
        if start is None:
            start = self.player.point
        if end is None:
            end = self.goal
        # Get the best path to the goal
        path: list[Point] = self.distance(start, end)
        # Compute the best direction based on the best path
        delta_dist: Point = path[1] - start
        return Point(np.sign(delta_dist.x), 0).asdirection() if abs(delta_dist.x) == self.n - 1 else delta_dist.asdirection()
    
    def move(self, dir: Direction) -> None:
        """Moves the player in the direction given"""
        # Reset the tile on the grid incase we move the player
        self.grid[self.player.y][self.player.x] = self.EMPTY if not self.on_goal() else self.GOAL

        # Check if the player is allowed to move in that direction
        match int(dir):
            case int(Direction.UP):
                if self._tile(self.player, offset_y=-1) != self.WALL:
                    self.player.move(dir)
            case int(Direction.DOWN):
                if self._tile(self.player, offset_y=1) != self.WALL:
                    self.player.move(dir)
            case int(Direction.LEFT):
                # Special check if the player is at the edge of the screen to wrap them to the otherside
                if self.player.x == 0 or self._tile(self.player, offset_x=-1) != self.WALL:
                    self.player.move(dir)
            case int(Direction.RIGHT):
                if self.player.x == self.n - 1 or self._tile(self.player, offset_x=1) != self.WALL:
                    self.player.move(dir)
        
        # Ensure the player is within the bounds of the arena
        if self.player.x >= self.n:
            self.player.x = 0
        elif self.player.x < 0:
            self.player.x = self.n - 1
        elif self.player.y >= self.m:
            self.player.y = 0
        elif self.player.y < 0:
            self.player.y = self.m - 1
        
        # Update the grid to display the players location
        self.grid[self.player.y][self.player.x] = self.PLAYER

    def set_goal(self) -> None:
        """Change the location of the goal. Should ony be called after Arena.on_goal() returns True"""
        # Clear the goal from the grid
        if self.on_goal():
            self.grid[self.player.y][self.player.x] = self.PLAYER
            self.player.collect_goal()
        else:
            self.grid[self.goal.y][self.goal.x] = self.EMPTY

        # Keep generating random locations for the goal while they are not walls
        tmp: Point = Point(0, 0)
        # Special condition: 2 tile cannot be reached an must be manually excluded
        while self._tile(tmp) == self.WALL or (tmp.x == 11 and (tmp.y == 5 or tmp.y == 17)):
            tmp.x = np.random.randint(0, 23)
            tmp.y = np.random.randint(1, 22)
        self.goal = tmp
        self.grid[self.goal.y][self.goal.x] = self.GOAL
    
    def detection(self) -> np.ndarray:
        """Returns the open space in the directions North, East, South, West"""
        # Scan out from the player in a direction until you hit a wall

        dist_up: int = 0
        while self._tile(self.player, offset_y=-dist_up) != self.WALL:
            dist_up += 1
        
        dist_down: int = 0
        while self._tile(self.player, offset_y=dist_down) != self.WALL:
            dist_down += 1

        tmp_pnt: Point = self.player.point.copy()
        dist_left: int = 0
        while self._tile(tmp_pnt, offset_x=-dist_left) != self.WALL:
            dist_left += 1
            if tmp_pnt.x - dist_left < 0:
                tmp_pnt.x = 23
        
        tmp_pnt: Point = self.player.point.copy()
        dist_right: int = 0
        while self._tile(tmp_pnt, offset_x=dist_right) != self.WALL:
            dist_right += 1
            if tmp_pnt.x + dist_right >= self.n:
                tmp_pnt.x = 0
    
        return np.array([dist_up, dist_right, dist_down, dist_left], np.float64) - np.ones(4)
    
    # The absolute distance from the player to the goal
    def absolute_distance(self) -> float:
        return math.sqrt((self.player.x - self.goal.x) ** 2 + (self.player.y - self.goal.y) ** 2)
    
    # The distance between two points in the grid using the A* algorithm
    def distance(self, start: Point | None = None, end: Point | None = None) -> list[Point] | None:
        global log
        # Set defaults for the start and end positions
        if start is None:
            start = self.player.point
        if end is None:
            end = self.goal

        pathKey: PathPair = PathPair(start, end)
        assert start is not None and end is not None and pathKey is not None
        if pathKey in self.path.keys():
            return self.path[pathKey]
        log.warning(f'  simulation.Arena: Distance cache miss {pathKey}')

        # Define function to calculate neighbors of points
        def neighbors(p: Point, arena: 'Arena' = self) -> list[Point]:
            # All direct neighbor points
            points: list[Point] = [Direction(i).topoint() for i in range(4)]
            ret: list[Point] = []
            for _p in points:
                # If point is inbounds and not a wall, add it to the list
                new_p: Point = _p + p

                if new_p.x == self.n:
                    new_p.x = 0
                elif new_p.x < 0:
                    new_p.x = self.n - 1
                elif new_p.y == self.m:
                    new_p.y = 0
                elif new_p.y < 0:
                    new_p.y = self.m - 1

                # If the point is inbounds add it to the list of valid neighbors
                if arena.grid[new_p.y][new_p.x] != Arena.WALL:
                    ret.append(new_p)
            return ret

        tmp = astar.find_path(start, end, neighbors_fnct=neighbors)
        # Convert iterator to list of points
        if tmp is not None:
            tmp = [p for p in tmp]
        # Update the path cache
        self.path[pathKey] = tmp
        log.debug(f'  Computed path from {start} to {end} as: {tmp}')
        return tmp
    
    # Textual representation of the grid
    def __str__(self) -> str:
        s: str = ""
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y] == self.EMPTY:
                    s += " "
                elif self.grid[x][y] == self.WALL:
                    s += "#"
                elif self.grid[x][y] == self.PLAYER:
                    s += "P"
                elif self.grid[x][y] == self.GOAL:
                    s += "G"
            s += "\n"
        return s

    # Converts the grid into a 2D texture with the given colors and block size
    def display(self, block_size: int = 10, wall_color: list[float] | None = None, player_color: list[float] | None = None, goal_color: list[float] | None = None) -> np.ndarray:
        COLOR_DEPTH: int = 4
        if wall_color is None:
            wall_color = [0, 0, 255 / 255, 255 / 255]
        if player_color is None:
            player_color = [255 / 255, 255 / 255, 0, 255 / 255]
        if goal_color is None:
            goal_color = [0, 255 / 255, 0, 255 / 255]

        wall_color:   np.ndarray = np.array(wall_color, dtype=np.float32)
        player_color: np.ndarray = np.array(player_color, dtype=np.float32)
        goal_color:   np.ndarray = np.array(goal_color, dtype=np.float32)

        if max(wall_color) > 255:
            wall_color /= 255
        if max(player_color) > 255:
            player_color /= 255
        if max(goal_color) > 255:
            goal_color /= 255
        
        texture_data: np.ndarray = np.zeros((self.n * block_size, self.m * block_size, COLOR_DEPTH))
        PATH_COLOR: np.ndarray = np.array([255 / 255, 0, 0, 255 / 255], dtype=np.float32)
        _map: dict[int, list[float]] = {
            Arena.EMPTY: np.array([0, 0, 0,0], dtype=np.float32), # Black
            Arena.WALL: wall_color, # Blue
            Arena.PLAYER: player_color, # Yellow
            Arena.GOAL: goal_color, # Green
        }
        path: list[Point] = self.distance()

        for y in range(self.m):
            for oy in range(block_size):
                for x in range(self.n):
                    for ox in range(block_size):
                        coord_y: int = y * block_size + oy
                        coord_x: int = x * block_size + ox
                        if Point(x, y) in path and self.grid[y][x] == Arena.EMPTY and (ox > 2 and ox < 8) and (oy > 2 and oy < 8):
                            texture_data[coord_y][coord_x] = PATH_COLOR
                        else:
                            texture_data[coord_y][coord_x] = _map[self.grid[y][x]]
        texture_data *= 255
        texture_data = texture_data.astype(np.uint8)
        return texture_data

def main():
    while key != 'q':
        print(arena)
        #arena.display()
        print(f"Player position: ({arena.player.x}, {arena.player.y})")
        print(f"Goal position: ({arena.goal.x}, {arena.goal.y})")
        key = input().lower()
        if key == 'w':
            arena.move(Direction.UP)
        elif key == 's':
            arena.move(Direction.DOWN)
        elif key == 'a':
            arena.move(Direction.LEFT)
        elif key == 'd':
            arena.move(Direction.RIGHT)
        
        if arena.on_goal():
            print("Player has reached the goal")
            arena.set_goal()
            print(f"Goal is now located at: ({arena.goal.x}, {arena.goal.y})")

if __name__ == "__main__":
    import sys
    import os
    import concurrent.futures
    import threading
    import pathlib
    import multiprocessing
    n = 23 # X length
    m = 23 # Y length
    arena = Arena(n, m)
    grid = arena.grid

    # Generate all points in the grid
    points: list[Point] = []
    for x in range(n):
        for y in range(m):
            points.append(Point(x, y))
    # Filter out all the points that the agent can go to
    points = list(filter(lambda x: grid[x.y][x.x] != Arena.WALL, points))

    lock: threading.Lock = threading.Lock()
    path_count = multiprocessing.Value('I', 0)
    def generate_path(start, end) -> list[Point] | None:
        global path_count
        tmp = arena.distance(start, end)
        with lock:
            path_count.value += 1
            print(f'Computed Path: {path_count.value}', end='\r')
        return tmp

    paths: dict[PathPair, list[Point] | None | concurrent.futures.Future[list[Point]]] = {}
    max_len: int = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for start in points:
            for end in points:
                future = executor.submit(generate_path, start, end)
                paths.setdefault(PathPair(start, end), future)

    p = {}
    for k, f in paths.items():
        result = f.result()
        max_len = max(len(result if result is not None else []), max_len)

        if result is not None:
            p[str(k)] = [i.__json__() for i in result]
        else:
            p[str(k)] = None
    print(f"Longest path of {len(paths)} paths is: {max_len}")

    # Export the generated paths to the cache file
    import json
    def default(o):
        if hasattr(o, '__json__'):
            return o.__json__()
        if isinstance(o, int):
            return int(o)
        return repr(o)

    with open(pathlib.Path(__file__).parent.parent / 'path_cache.json', 'w') as f:
        json.dump(p, f)

    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        main()