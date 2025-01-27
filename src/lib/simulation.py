#!/usr/bin/env python3

import numpy as np
import math
from enum import IntEnum

class Point:
    def todistance(self, dtype: np.dtype = np.float64) -> np.ndarray:
        tmp: np.ndarray = np.array([-self.y, self.x, self.y, -self.x], dtype=dtype)
        return np.maximum(tmp, np.zeros(tmp.shape))

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x or self.y < other.y
    
    def __gt__(self, other: 'Point') -> bool:
        return self.x > other.x or self.y > other.y
    
    def __eq__(self, value: 'Point') -> bool:
        return self.x == value.x and self.y == value.y
    
    def __hash__(self):
        return hash((self.x - self.y) * (self.x * self.y))

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    def __json__(self) -> dict[str]:
        return {
            'x': self.x,
            'y': self.y
        }
        
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Player:
    @classmethod
    def frompoint(cls, p: Point) -> 'Player':
        return cls(p.x, p.y, 0, [])
    
    @classmethod
    def fromcoordinate(cls, x: int, y: int) -> 'Player':
        return cls(x, y, 0, [])

    def __init__(self, x: int, y: int, score: int, positions: list[Point]):
        self.point: Point = Point(x, y)
        self.score: int = score
        self.positions: list[Point] = positions
    
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
    
    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

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

    @property
    def x(self) -> int:
        return self.point.x
    
    @x.setter
    def x(self, x: int) -> None:
        self.point.x = x
    
    @property
    def y(self) -> int:
        return self.point.y
    
    @y.setter
    def y(self, y: int) -> None:
        self.point.y = y

class Arena:
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    GOAL = 3

    _player_start: Point = Point(3,3)
    _goal_start: Point = Point(6, 1)
    
    def __init__(self, n: int = 23, m: int = 23):
        self.n: int = n
        self.m: int = m
        self.player: Player = Player.frompoint(Arena._player_start)
        self.goal: Point = Point(Arena._goal_start.x, Arena._goal_start.y)
        self.grid: np.ndarray = Arena._create_grid()
    
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
                if self.player.x == 22 or self._tile(self.player, offset_x=1) != self.WALL:
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
        
        # Add the updated position to the list of player positions
        self.player.positions.append(Point(self.player.x, self.player.y))
        
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
        dist_up = 0
        while self._tile(self.player, offset_y=dist_up-1) != self.WALL:
            dist_up += 1

        dist_down = 0
        while self._tile(self.player, offset_y=dist_down+1) != self.WALL:
            dist_down += 1

        dist_left = 0
        wrap_offset_left = 0
        while self._tile(self.player, offset_x=dist_left-1) != self.WALL:
            dist_left += 1

        dist_right = 0
        wrap_offset_right = 0
        while self._tile(self.player, offset_x=dist_right+1) != self.WALL:
            dist_right += 1

        return np.array([dist_up, dist_right, dist_down, dist_left], np.float64)
    
    # The absolute distance from the player to the goal
    def distance(self) -> float:
        return math.sqrt((self.player.x - self.goal.x) ** 2 + (self.player.y - self.goal.y) ** 2)
    
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

    def display(self, block_size: int = 10, wall_color: int = 0x0000FFFF, player_color: int = 0xFFFF00FF, goal_color: int = 0x00FF00FF) -> np.ndarray:
        pass

if __name__ == "__main__":
    n = 23 # X length
    m = 23 # Y length
    arena = Arena(n, m)
    grid = arena.grid
    key: str = ""

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