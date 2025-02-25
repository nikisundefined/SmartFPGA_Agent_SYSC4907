import nengo.solvers
import nengo.neurons
import nengo.learning_rules
import logging
import json
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from simulation import Direction, Arena, Player, Point
from dataclasses import dataclass, asdict

# Default json encoder for custom object serialization
class JsonEncoder(json.JSONEncoder):
    def __init__():
        super()
    
    def default(self, o):
        try:
            return o.__json__()
        except:
            pass
        return super().default(o)

# Create a property with the given name and type
def create_property(name: str, attr_type: type) -> property:
    def getter(self):
        return getattr(self, f"_{name}")

    def setter(self, value) -> None:
        setattr(self, f"_{name}", value)
    
    def deleter(self) -> None:
        delattr(f"_{name}")

    return property(getter, setter, deleter, f"Property for {name}: {attr_type}")

# AttrDict Style Guideline
#   Access to get/set variables should be through attribute: ex. cvars.learning_rate / cvars.learning_rate = x
#   Field access function should be overridable for subclasses: ex. SubConsoleDict(ConsoleDict) -> def get_learning_rate(self)
#   Simple definition of variables contained within structure with type hints: ex. dataclass style

# Default values for ConsoleDict Variables
@dataclass(frozen=True)
class DefaultConsoleDict:
    # The last action performed by the agent in the simulation
    last_action: Direction = Direction.UP
    # Did the player's location change from the last action
    player_moved: bool = True
    # The number of neurons per ensemble
    ensemble_neurons: int = 1000
    # Learning rate of the learning rule
    learning_rate: float = 5e-5
    # The adaptive factor used with the Adaptive LIF neuron type
    tau_n: float = 0.01
    # Neuron type used in all ensembles
    neuron_type: nengo.neurons.NeuronType = nengo.neurons.SpikingRectifiedLinear()
    # Solver type used for learning connections
    solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=True)
    # Learning rule used for learning connections
    learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate, pre_synapse=None)
    # Number of dimensions input to the model
    input_dimensions: int = 4
    # Number of dimensions output from the model
    output_dimensions: int = 4
    # Number of dimensions ouput from the error function
    error_dimensions: int = 4
    # The datatype used for all numpy arrays
    dtype: np.dtype = np.float16
    # The current reward of the agent
    reward: float = 1.0
    # The minimum value for an action to be selected
    movement_threshold: float = 1e-6
    # The arena the agent with move within
    arena: Arena = Arena()
    # Was an action performed by the agent since the last action
    action_performed: bool = False
    # Flag if the current action moved away from the goal
    moved_away_from_goal: int = 0
    # The synapse used for the connection between pre and post ensembles
    connection_synapse: float = 0.01
    # The logging level for the script to print out
    log_level: int = int(logging.DEBUG)
    # Path to the A* path cache file
    path_cache_file: Path = Path(__file__).parent.parent / 'path_cache.json'
    # Selector for the alternate set of inputs to the model (Only useful with the fpga model)
    alt_input: bool = True
    # Should the reward reset after collecting a goal
    reward_reset: bool = False

# Access classs for Console Variables with local storage
class ConsoleDict:
    def __init__(self) -> None:
        default_dict: dict[str] = asdict(DefaultConsoleDict())
        default_dict = {'_' + k: v for k, v in default_dict.items()}
        self.__dict__.update(default_dict)
# Load all properties from the defaults for easy access
for attr, type in DefaultConsoleDict.__annotations__.items():
        setattr(ConsoleDict, attr, create_property(attr, type))
    
# Stores variables related to shared implementation of variable storage
@dataclass
class SharedDict:
    # Name of the shared memory that contains the cvars structure
    cvars_name: str = 'cvars'
    # Name of the shared memory that contains the gvars structure
    gvars_name: str = 'gvars'
    player_name: str = 'agent'
    arena_name: str = 'arena'

# Store variables related to the GUI implementation that should be shared between all GUI implementations
@dataclass(frozen=True)
class DefaultGUIDict:
    # The arena the GUI representation should be based from
    arena: Arena = None
    # Is the local gui running
    in_gui: bool = False
    # The start time of the current simulation
    start_time: float = 0.0
    # The current seed of the simulation
    seed: int = -1

# Stores all variables related to the local and nengo gui representation
class GUIDict:
    def __init__(self) -> None:
        default_dict: dict[str] = asdict(DefaultGUIDict())
        default_dict = {'_' + k: v for k, v in default_dict.items()}
        self.__dict__.update(default_dict)
# Load all properties from the defaults for easy access
for attr, type in DefaultGUIDict.__annotations__.items():
        setattr(GUIDict, attr, create_property(attr, type))

cvar: ConsoleDict = ConsoleDict()
gvar: GUIDict = GUIDict()

if __name__ == '__main__':
    ### Global initialization
    import shared
    _log: logging.Logger = logging.getLogger(__file__)

    if 'svar' not in globals():
        global svar
        svar: SharedDict = SharedDict()
        _log.debug('Initialized shared variables')

    # If console vars are not loaded, attempt to load them from shared memory
    if 'cvar' not in globals():
        _log.debug('cvar not found, attempting to load from shared memory')
        # global cvar
        try:
            tmp = SharedMemory(name=svar.cvars_name, create=False)
            _log.debug(f'Loaded Console Variables from shared memory with name: {svar.cvars_name}')
        except:
            _log.warning('Could not load Console Variables from shared memory, creating new instance')
            tmp = SharedMemory(name=svar.cvars_name, create=True, size=shared.SharedConsoleDict.size)
        cvar: ConsoleDict = shared.SharedConsoleDict(tmp.buf)
        
    # If GUI vars are not loaded, attempt to load them from shared memory
    if 'gvar' not in globals():
        _log.debug('gvar not found, attempting to load from shared memory')
        # global gvar
        try:
            tmp = SharedMemory(name=svar.gvars_name, create=False)
            _log.debug(f'Loaded GUI Variables from shared memory with name: {svar.gvars_name}')
        except:
            _log.warning('Could not load GUI Variables from shared memory, creating new instance')
            tmp = SharedMemory(name=svar.gvars_name, create=True, size=shared.SharedGUIDict.size)
        gvar: GUIDict = shared.SharedGUIDict(tmp.buf)
        gvar.arena = cvar.arena