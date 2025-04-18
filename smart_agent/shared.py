#!/usr/bin/env python3

import multiprocessing.shared_memory as shm
import numpy as np
import atexit
import ctypes
import logging
import inspect
import smart_agent.simulation as simulation
from enum import IntEnum

# Import class names into namespace, needed due to cyclic imports
Arena = simulation.Arena
Player = simulation.Player
Point = simulation.Point
Direction = simulation.Direction
PathCache = simulation.PathCache
PathPair = simulation.PathPair
PlayerInfo = simulation.PlayerInfo
Performance = simulation.Performance 

"""Shared Implementaion and Proxy classes for the variable storage classes defined in vars.py and simulation.py"""

# Storeable types:
#   simulation.Point -> SharedPoint -> int32 * 2
#   simulation.Player -> SharedPlayer -> uint32 + SharedPoint
#   simulation.Arena -> SharedArena -> uint8 * n * m + uint32 + uint32 + SharedPlayer + SharedPoint
#   simulation.PathCache -> SharedPathCache
#   simulation.Direction -> uint8
#   simulation.PathPair -> (SharedPoint, SharedPoint)
#   int -> int32
#   float -> double
#   bool -> uint8
#   str -> uint8[255]

import smart_agent.vars as vars
log: logging.Logger = logging.getLogger('smart_agent.shared')

# Store the line terminators for all handlers to allow printing on the same line
def push_log_terminiator() -> dict[logging.Handler, str]:
    ret: dict[logging.Handler, str] = {}
    for h in logging.getLogger('smart_agent').handlers:
        if isinstance(h, logging.StreamHandler):
            ret.setdefault(h, h.terminator)
            h.terminator = ""
    return ret

# Restore the line handlers that were previously saved
def pop_log_terminator(terms: dict[logging.Handler, str]) -> None:
    for h in logging.getLogger('smart_agent').handlers:
        if h in terms:
            h.terminator = terms[h]

# Setup the collection of all shared memory segments to cleanup on exit
if 'shm_names' not in globals():
    global shm_names
    shm_names: dict[str, tuple[shm.SharedMemory, bool, str]] = {}

    # Cleanup function that removes all
    def cleanup():
        global shm_names
        for name, pair in shm_names.items():
            # Try to release the resource
            try:
                pair[0].buf.release()
                pair[0].close()
            except Exception as e:
                log.warning(f'Failed to close shared memory region {pair[0].name}\n{e}\nBacktrace:\n{pair[2]}')
            
            # If we should remove the resource, attempt to do so
            if pair[1]:
                try:
                    pair[0].unlink()
                except Exception as e:
                    log.warning(f'Failed to unlink shared memory region {pair[0].name}\n{e}')
                    
    # Ensure the cleanup handler has been registered for atexit
    try:
        atexit.unregister(cleanup)
    except:
        pass
    atexit.register(cleanup)

# Enum to define how to attach to shared memory
class AttachFlag(IntEnum):
    DONT_ATTACH = 0
    ATTACH = 1
    ATTACH_AND_CLEANUP = 2

# Create a region of shared memory with the given size
#   Optionally, can be named to be reused later with the same name
#   Optionally, attempt to attach to the given name
def create_shared_memory(size: int, name: str = None, attach: AttachFlag = AttachFlag.DONT_ATTACH) -> memoryview:
    global shm_names
    if attach > int(AttachFlag.DONT_ATTACH):
        assert name is not None, "Name cannot be None when trying to attach to shared memory region"
    ret = shm.SharedMemory(name=name, create=(attach == AttachFlag.DONT_ATTACH), size=size)
    tmp = ""
    count: int = 0
    # Generate stack frame information for debug tracking of shared memory regions
    for frame in inspect.stack():
        tmp += f"{count}: {frame.filename}.{frame.function}:{frame.lineno}\n"
        count += 1
    # If a new block of memory was allocated print out the entire stack
    if attach == int(AttachFlag.DONT_ATTACH):
        log.debug(f'Allocated {size} bytes with name {ret.name} for:\n{tmp}')
    # Else, print out the function that attached to the given memory block
    else:
        tmpl = inspect.stack()[1]
        log.debug(f'Allocated {size} bytes with name {ret.name} for {tmpl.filename}.{tmpl.function}')
    shm_names.setdefault(ret.name, (ret, attach != AttachFlag.ATTACH, tmp))
    return ret.buf

# Creates a property that stores the value in shared memory given by the memoryview object
def create_shared_property(name: str, attr_type: type) -> property:
    def getter(self):
        return attr_type(getattr(self, f'_{name}').value)
    def setter(self, value):
        if not isinstance(value, attr_type):
            raise TypeError(f'{type(value)} is not assignable to {attr_type} for attribute {name}')
        getattr(self, f'_{name}').value = value
    def deleter(self):
        delattr(self, f'_{name}')
    return property(getter, setter, deleter)

# A point that is stored in shared memory
class SharedPoint(Point):
    # The number of bytes required for one object
    size: int = ctypes.sizeof(ctypes.c_int32) * 2

    # Create an empty shared point with its own shared memory space
    @classmethod
    def create_shared_point(cls, name: str | None = None) -> 'SharedPoint':
        return cls(create_shared_memory(SharedPoint.size, name=name))

    # Convert a Point into a shared point leaving the original untouched
    @classmethod
    def frompoint(cls, p: Point) -> 'SharedPoint':
        tmp = create_shared_memory(SharedPoint.size)
        tmp = cls(tmp)
        tmp.x = p.x
        tmp.y = p.y
        return tmp
    
    @classmethod
    def fromvector(cls, vec: np.ndarray) -> 'SharedPoint':
        assert vec.shape == (4,), f"Invalid shape for vector: {vec.shape}"
        p: Point = Point(vec[1] - vec[3], vec[2] - vec[0])
        return cls.frompoint(p)
    
    # Convert a list of points into a list of shared points
    # Allocates a large block of shared memory instead of small blocks for each point
    @classmethod
    def convertpointlist(cls, points: list[Point], buf: memoryview | None = None) -> list['SharedPoint']:
        size = len(points) * SharedPoint.size # nbytes of shared memory required for this list
        if buf is not None: # If a buffer was given
            if buf.nbytes < size: # Ensure it has enough space to store all the points
                raise BufferError(f'Not enough bytes in buffer for te given list: {buf.nbytes} < {size}')
            tmp = buf
        else: # Else create the shared memory buffer to be used
            tmp = create_shared_memory(size)
        shared_points = []
        for p, offset in zip(points, range(0, size, SharedPoint.size)):
            sp = cls(tmp[offset:offset + SharedPoint.size])
            shared_points.append(sp)
            sp.assign(p)
        return shared_points
    
    # Create a deep copy of the current point-like object
    def copy(self) -> 'SharedPoint':
        tmp = SharedPoint.create_shared_point()
        tmp.x = self.x
        tmp.y = self.y
        return tmp
    
    # Create a shared clone of the current point-like object
    def clone(self) -> 'SharedPoint':
        return SharedPoint(self.buf)

    def __init__(self, buf):
        if isinstance(buf, SharedPoint):
            buf = buf.buf
        assert buf.nbytes >= 8, "Not enough space in the given buffer"
        self.buf: memoryview = buf
        self._x = ctypes.c_int32.from_buffer(buf[:4])
        self._y = ctypes.c_int32.from_buffer(buf[4:])

    def __iadd__(self, other: Point | Player | Direction) -> None:
        if type(other) is Direction:
            other = other.topoint()
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other: Point | Player | Direction) -> None:
        if type(other) is Direction:
            other = other.topoint()
        self.x -= other.x
        self.y -= other.y
        return self

    # Assigns the value of another point to this point object
    def assign(self, other: Point) -> None:
        self.x = other.x
        self.y = other.y

    @property
    def x(self) -> int:
        return self._x.value
    
    @x.setter
    def x(self, x: int) -> None:
        self._x.value = x

    @property
    def y(self) -> int:
        return self._y.value
    
    @y.setter
    def y(self, y: int) -> None:
        self._y.value = y

class SharedPathCache(PathCache):
    size: int = -1
    
    @staticmethod
    def permute_keys(arena: Arena) -> list[Point]:
        n: int = arena.n
        m: int = arena.m
        grid: np.ndarray = arena.grid
        # Generate all points in the grid
        points: list[Point] = []
        for x in range(n):
            for y in range(m):
                points.append(Point(x, y))
        # Filter out all the points that the agent can go to
        points = list(filter(lambda x: grid[x.y][x.x] != Arena.WALL, points))
        return points

    def __init__(self, buf: memoryview):
        self.buf: memoryview = buf
        self.paths: dict[PathPair, list[SharedPoint] | None] = {}
        self.cache = self.paths

    # Loads this SharedPathCache object with data contained within the given PathCache or within 
    def load(self, cache: PathCache) -> None:
        # If a cache was given
        if cache is not None:
            # Compute how many bytes of shared memory are needed to store the cache
            nbytes: int = 0
            for k, v in cache.cache.items():
                nbytes += 0 if v is None else len(v) * SharedPoint.size
            # If there is enough space in the internal buffer, do not allocate a new one
            if self.buf is None or self.buf.nbytes < nbytes:
                log.warning(f"Attempted to load SharedPathCache object with insufficient space in internal buffer: {0 if self.buf is None else self.buf.nbytes} < {nbytes}")
                self.buf = create_shared_memory(nbytes)
        offset: int = 0
        count: int = 0
        self.paths.clear() # Reset the internal mapping of points
        tmp = push_log_terminiator()
        for k, v in cache.cache.items():
            nbytes = 0 if v is None else len(v) * SharedPoint.size 
            self.paths.setdefault(k, v if v is None else SharedPoint.convertpointlist(v, self.buf[offset:offset+nbytes]))
            offset += nbytes
            log.debug(f'Loaded path {count}    \r')
            count += 1
        pop_log_terminator(tmp)

    # Returns a key mappping value to load data from the backing memoryview without making a copy
    def keys(self) -> dict[PathPair, int]:
        return {k: 0 if v is None else len(v) for k, v in self.paths.items()}

    # Loads the cache from the internal buf with the given keys and lengths
    def loadkeys(self, keys: dict[PathPair, int] | memoryview) -> None:
        if type(keys) is memoryview:
            raise NotImplementedError()

        count: int = sum(keys.values())
        if self.buf.nbytes < count * SharedPoint.size:
            raise BufferError("Internal buffer does not contain enough information for the given set of keys")
        
        def create_list(buf: memoryview) -> list[SharedPoint]:
            assert buf.nbytes % SharedPoint.size == 0, "Buffer size is unaligned"
            ret: list[SharedPoint] = []
            for i in range(0, buf.nbytes, SharedPoint.size):
                ret.append(SharedPoint(buf[i:i+SharedPoint.size]))
            return ret

        self.cache.clear()
        offset: int = 0
        for k, v in keys.items():
            self.paths.setdefault(k, None if v == 0 else create_list(self.buf[offset:offset + v * SharedPoint.size]))
            offset += v * SharedPoint.size

    # Serializes this PathCache object into JSON
    def __json__(self) -> dict[str, dict[str, int] | None]:
        return {str(k): None if v is None else [e.__json__() for e in v] for k, v in self.paths.items()}

    def __setitem__(self, pair, path):
        raise RuntimeError(f"SharedPathCache is immutable")

class SharedPlayerInfo(PlayerInfo):
    size: int = ctypes.sizeof(ctypes.c_uint32) * 2 + ctypes.sizeof(ctypes.c_double)

    def __init__(self, buf: memoryview):
        self._buf: memoryview = buf
        self._actions: ctypes.c_uint32 = ctypes.c_uint32.from_buffer(buf[:4])
        self._time: ctypes.c_uint32 = ctypes.c_uint32.from_buffer(buf[4:8])
        self._reward: ctypes.c_double = ctypes.c_double.from_buffer(buf[8:])
    
    def copy(self, buf: memoryview) -> 'SharedPlayerInfo':
        tmp = SharedPlayerInfo(buf)
        tmp.actions = self.actions
        tmp._time = self._time
        tmp._reward = self._reward
        return tmp
    
    def clone(self) -> 'SharedPlayerInfo':
        return SharedPlayerInfo(self._buf)
    
    @property
    def actions(self) -> int:
        return self._actions.value
    
    @actions.setter
    def actions(self, other: int) -> None:
        self._actions.value = other

    @property
    def time(self) -> int:
        return self._time.value
    
    @time.setter
    def time(self, other: int) -> None:
        self._time.value = other

    @property
    def reward(self) -> float:
        return self._reward.value
    
    @reward.setter
    def reward(self, other: float) -> float:
        self._reward.value = other

class SharedPlayer(Player):
    size: int = ctypes.sizeof(ctypes.c_uint32) + SharedPoint.size

    @classmethod
    def fromsharedpoint(cls, p: SharedPoint, name: str | None = None) -> 'SharedPlayer':
        tmp = create_shared_memory(name=name, size=SharedPlayer.size)
        tmp = cls(tmp)
        tmp.point.assign(p)
        return tmp
    
    def __init__(self, buf):
        if isinstance(buf, SharedPlayer):
            buf = buf._buf
        self._buf: memoryview = buf
        self._score: ctypes.c_uint32 = ctypes.c_uint32.from_buffer(buf[:ctypes.sizeof(ctypes.c_uint32)])
        self.point: SharedPoint = SharedPoint(buf[ctypes.sizeof(ctypes.c_uint32):ctypes.sizeof(ctypes.c_uint32) + SharedPoint.size])
        self.positions: list[Point] = []
        self.info: PlayerInfo = PlayerInfo(0, 0, 0)
    
    def copy(self) -> 'SharedPlayer':
        tmp = SharedPlayer.fromsharedpoint(self.point)
        tmp.positions = self.positions.copy()

    def clone(self) -> 'SharedPlayer':
        tmp = SharedPlayer(self._buf)
        tmp.positions = self.positions.copy()
        return tmp
    
    def move(self, dir: Direction) -> None:
        self.point += Direction(dir)
    
    @property
    def score(self) -> int:
        return self._score.value
    
    @score.setter
    def score(self, value: int) -> None:
        self._score.value = value

class SharedArena(Arena):
    size: int = Arena._create_grid().nbytes + ctypes.sizeof(ctypes.c_uint32) * 2 + SharedPlayer.size + SharedPoint.size
    shared_path_cache: SharedPathCache = SharedPathCache(None)

    def __init__(self, n: int = 23, m: int = 23, buf: memoryview | None = None):
        # uint8 * n * m + uint32 + uint32 + SharedPlayer + SharedPoint
        self.size: int = ctypes.sizeof(ctypes.c_uint8) * n * m + ctypes.sizeof(ctypes.c_uint32) * 2 + SharedPlayer.size + SharedPoint.size
        if buf is None:
            self.buf = create_shared_memory(self.size)
        elif buf.nbytes < self.size:
            raise BufferError("Not enough space in buffer for SharedArena Object")
        else:
            self.buf = buf
        offset: int = 0

        self.n: int = n
        self.m: int = m
        self.player: SharedPlayer = SharedPlayer(self.buf[:SharedPlayer.size])
        self.player.point.assign(Arena._player_start)
        offset += SharedPlayer.size
        self.goal: SharedPoint = SharedPoint(buf[offset:offset+SharedPoint.size])
        self.goal.assign(Arena._goal_start)
        offset += SharedPoint.size
        grid: np.ndarray = Arena._create_grid()
        self.grid: np.ndarray = np.ndarray(grid.shape, dtype=grid.dtype, buffer=self.buf[offset:offset+grid.nbytes])
        self.grid[:] = grid[:]

        self.performance = Performance()
    
    def distance(self, start: Point | None = None, end: Point | None = None) -> list[Point] | None:
        global log
        # Set defaults for the start and end positions
        if start is None:
            start = self.player.point
        if end is None:
            end = self.goal

        pathKey: PathPair = PathPair(start, end)
        pathKeyReverse: PathPair = PathPair(end, start)
        assert start is not None and end is not None and pathKey is not None
        # Check both the forward and backward paths, since lookups are cheaper than computation
        if pathKey in SharedArena.shared_path_cache:
            return SharedArena.shared_path_cache[pathKey]
        elif pathKeyReverse in SharedArena.shared_path_cache:
            return list(reversed(SharedArena.shared_path_cache[pathKeyReverse]))
        with Arena.path_cache_lock:
            if pathKey in Arena.path_cache:
                return Arena.path_cache[pathKey]
            elif pathKeyReverse in Arena.path_cache:
                return list(reversed(Arena.path_cache[pathKeyReverse]))
        log.warning(f'Entry not found in path cache: {pathKey}')
        tmp = self._distance(start, end)
        # If there is a valid path between the start and end points
        if tmp is not None:
            count: int = 0
            # Go over every sub-path in the orignal path and add them all to the cache
            for point in tmp:
                key: PathPair = PathPair(point, end)
                Arena.path_cache[key] = tmp[count:]
                count += 1
        else: # There is not viable path, update the cache with None value
            Arena.path_cache[pathKey] = tmp
        log.debug(f'Computed {len(tmp)} paths from {start} to {end} as: {tmp}')
        return tmp

# Helper function to copy the default value from a dataclass and convert the internal field into a shared varient
def copy_default_params_to_shared(self, base_class: type):
    offset: int = 0
    for k, v in vars.asdict(base_class()).items():
        internal_name: str = f'_{k}'
        attr_type: type = type(getattr(self, internal_name))
        if issubclass(attr_type, bool):
            setattr(self, internal_name, ctypes.c_uint8.from_buffer(self.buf[offset:offset+ctypes.sizeof(ctypes.c_uint8)]))
            offset += ctypes.sizeof(ctypes.c_uint8)
            getattr(self, internal_name).value = bool(v)
        elif issubclass(attr_type, int):
            setattr(self, internal_name, ctypes.c_int32.from_buffer(self.buf[offset:offset+ctypes.sizeof(ctypes.c_int32)]))
            offset += ctypes.sizeof(ctypes.c_int32)
            getattr(self, internal_name).value = int(v)
        elif issubclass(attr_type, float):
            setattr(self, internal_name, ctypes.c_double.from_buffer(self.buf[offset:offset+ctypes.sizeof(ctypes.c_double)]))
            offset += ctypes.sizeof(ctypes.c_double)
            getattr(self, internal_name).value = float(v)
        elif issubclass(attr_type, str):
            val: bytes = str(v).encode()
            setattr(self, internal_name, self.buf[offset:offset+255])
            offset += 255
            getattr(self, internal_name)[:] = val[:]
        elif attr_type is Arena:
            setattr(self, internal_name, SharedArena(buf=self.buf[offset:offset+SharedArena.size]))
            offset += SharedArena.size

# Should effectivly emulate ConsoleDict, but stores its values in a named shared memory buffer
class SharedConsoleDict(vars.ConsoleDict):
    # Size of allocated shared memory for the data structure
    size: int = 0

    def __init__(self, buf: memoryview):
        super().__init__()
        self.buf = buf
        copy_default_params_to_shared(self, vars.DefaultConsoleDict)
        
# Helper function to convert local storage properties of dataclasses into shared properties
def attach_shared_property(base_class: type, shared_class: type):
    for _attr, _type in base_class.__annotations__.items():
        if issubclass(_type, bool):
            setattr(shared_class, _attr, create_shared_property(_attr, bool))
            shared_class.size += ctypes.sizeof(ctypes.c_uint8)
        elif issubclass(_type, int):
            setattr(shared_class, _attr, create_shared_property(_attr, int))
            shared_class.size += ctypes.sizeof(ctypes.c_int32)
        elif issubclass(_type, float):
            setattr(shared_class, _attr, create_shared_property(_attr, float))
            shared_class.size += ctypes.sizeof(ctypes.c_double)
        elif issubclass(_type, str):
            def getter(self):
                return str(getattr(self, _attr))
            def setter(self, value):
                raise NotImplementedError()
            setattr(shared_class, _attr, property(getter, setter))
            shared_class.size += 255
        elif _type is Arena:
            shared_class.size += SharedArena.size
attach_shared_property(vars.DefaultConsoleDict, SharedConsoleDict)

class SharedGUIDict(vars.GUIDict):
    size: int = 0

    def __init__(self, buf: memoryview):
        super().__init__()
        self.buf = buf
        copy_default_params_to_shared(self, vars.DefaultGUIDict)
attach_shared_property(vars.DefaultGUIDict, SharedGUIDict)

if __name__ == '__main__':
    sm = create_shared_memory(SharedConsoleDict.size)
    tmp = SharedConsoleDict(sm)

    try:
        sm = create_shared_memory(0, 'path_cache_json', True)
        s = bytes.decode(sm.tobytes())
    except:
        import pathlib
        s = pathlib.Path('path_cache.json').read_text()
        b = s.encode()
        sm = create_shared_memory(len(b), 'path_cache_json')
        sm[:] = b[:]
        del shm_names[0]
    tmp = simulation.PathCache.fromjson(s)
    nbytes: int = tmp.count() * SharedPoint.size
    attached: bool = False
    try:
        buf = create_shared_memory(nbytes, 'path_cache')
    except:
        buf = create_shared_memory(nbytes, 'path_cache', AttachFlag.ATTACH)
        attached = True
    spc = SharedPathCache(buf)
    if attached:
        spc.load(tmp)
    spc.loadkeys()