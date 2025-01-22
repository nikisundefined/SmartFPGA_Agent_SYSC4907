#!/usr/bin/env python3

import numpy as np
import random
import math
from PIL import Image, ImageDraw
from enum import IntEnum

goal_positions = [
    (6, 1),
    (16, 1),
    (15, 5),
]

class Point:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x or self.y < other.y
    
    def __gt__(self, other: 'Point') -> bool:
        return self.x > other.x or self.y > other.y
    
    def __eq__(self, value: 'Point') -> bool:
        return self.x == value.x and self.y == value.y

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"({self.x},{self.y})"
        
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Arena:
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    GOAL = 3

    _player_start: Point = Point(3, 3)
    _goal_start: Point = Point(6, 1)
    
    def __init__(self, n: int = 23, m: int = 23):
        self.n: int = n
        self.m: int = m
        self.player: Point = Point(Arena._player_start.x, Arena._player_start.y)
        self.goal: Point = Point(Arena._goal_start.x, Arena._goal_start.y)
        self.grid: np.ndarray = Arena._create_grid()
    
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
    
    def on_goal(self) -> bool:
        return self.player.x == self.goal.x and self.player.y == self.goal.y
    
    def move(self, dir: Direction) -> None:
        if type(dir) == int and dir < 4:
            dir = Direction(dir)
        elif type(dir) != Direction:
            raise TypeError()
        
        self.grid[self.player.y][self.player.x] = self.EMPTY if not self.on_goal() else self.GOAL

        if dir == int(Direction.RIGHT) and self.player.x == 22:
            self.player.x = 23
        elif dir == int(Direction.LEFT) and self.player.y == 0:
            self.player.x = 22
        elif dir == int(Direction.UP) and self.grid[self.player.y - 1][self.player.x] != self.WALL:
            self.player.y -= 1
        elif dir == int(Direction.RIGHT) and self.grid[self.player.y][self.player.x + 1] != self.WALL:
            self.player.x += 1
        elif dir == int(Direction.DOWN) and self.grid[self.player.y + 1][self.player.x] != self.WALL:
            self.player.y += 1
        elif dir == int(Direction.LEFT) and self.grid[self.player.y][self.player.x - 1] != self.WALL:
            self.player.x -= 1
        
        if self.player.x >= self.n:
            self.player.x = 0
        elif self.player.x < 0:
            self.player.x = self.n - 1
        elif self.player.y >= self.m:
            self.player.y = 0
        elif self.player.y < 0:
            self.player.y = self.m - 1
        
        self.grid[self.player.y][self.player.x] = self.PLAYER
    
    def set_goal(self) -> None:
        if self.on_goal():
            self.grid[self.player.y][self.player.x] = self.PLAYER
        else:
            self.grid[self.goal.y][self.goal.x] = self.EMPTY

        tmp_x: int = 0
        tmp_y: int = 0
        while self.grid[tmp_y][tmp_x] == self.WALL or (tmp_x == 11 and (tmp_y == 5 or tmp_y == 17)):
            tmp_x = np.random.randint(0, 23)
            tmp_y = np.random.randint(1, 22)
        self.goal.x = tmp_x
        self.goal.y = tmp_y
        self.grid[self.goal.y][self.goal.x] = self.GOAL
    
    def detection(self) -> np.ndarray:
        dist_up = 0
        while self.grid[self.player.y - dist_up - 1][self.player.x] != self.WALL:
            dist_up += 1
        dist_down = 0
        while self.grid[self.player.y + dist_down + 1][self.player.x] != self.WALL:
            dist_down += 1
        dist_left = 0
        while self.grid[self.player.y][self.player.x - dist_left - 1] != self.WALL:
            dist_left += 1
        dist_right = 0
        while self.grid[self.player.y][self.player.x + dist_right + 1] != self.WALL:
            dist_right += 1
        return np.array([dist_up, dist_down, dist_left, dist_right], np.float64)
    
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

    def display(self) -> None:
        pass # TODO: Output a PIL Image that is saved when this function is called

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
            new_goal = random.choice(list(set(goal_positions + [(arena.goal.x, arena.goal.y)])))
            arena.set_goal(new_goal[0], new_goal[1])
            print(f"Goal is now located at: ({arena.goal.x}, {arena.goal.y})")