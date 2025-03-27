import nengo.solvers
import nengo.neurons
import nengo.learning_rules
import logging
import json
import numpy as np
import pathlib
import smart_agent.simulation as simulation
from dataclasses import dataclass, asdict
import nengopy.neurons

Arena = simulation.Arena
Direction = simulation.Direction

log: logging.Logger = logging.getLogger('smart_agent.vars')

# Default json encoder for custom object serialization
class JsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default(self, o):
        try:
            return o.__json__()
        except AttributeError:
            pass
        return super().default(o)

# Create a property with the given name and type
def create_property(name: str, attr_type: type) -> property:
    def getter(self):
        return getattr(self, f"_{name}")

    def setter(self, value) -> None:
        setattr(self, f"_{name}", value)
    
    def deleter(self) -> None:
        delattr(self, f"_{name}")

    return property(getter, setter, deleter, f"Property for {name}: {attr_type}")

# AttrDict Style Guideline
#   Access to get/set variables should be through attribute: ex. cvars.learning_rate / cvars.learning_rate = x
#   Field access function should be overridable for subclasses: ex. SubConsoleDict(ConsoleDict) -> def get_learning_rate(self)
#   Simple definition of variables contained within structure with type hints: ex. dataclass style

# Default values for ConsoleDict Variables
@dataclass(frozen=True)
class DefaultConsoleDict:
    # The last action performed by the agent in the simulation
    last_action: Direction = Direction.NONE
    # Did the player's location change from the last action
    player_moved: bool = True
    # The number of neurons per ensemble
    ensemble_neurons: int = 400
    # Learning rate of the learning rule
    learning_rate: float = 5e-4
    # The adaptive factor used with the Adaptive LIF neuron type
    tau_n: float = 0.01
    # Neuron type used in all ensembles
    neuron_type: nengo.neurons.NeuronType = nengopy.neurons.RectifiedLinear()
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
    movement_threshold: float = 1e-5
    # The arena the agent with move within
    arena: Arena = Arena()
    # Was an action performed by the agent since the last action
    action_performed: bool = False
    # Flag if the current action moved away from the goal
    moved_away_from_goal: int = 0
    # The synapse used for the connection between pre and post ensembles
    connection_synapse: float = 0.01
    # The logging level for the script to print out
    log_level: int = int(logging.INFO)
    # Path to the A* path cache file
    path_cache_file: pathlib.Path = pathlib.Path(__file__).parent.parent / 'path_cache.json'
    # Selector for the alternate set of inputs to the model (Only useful with the fpga model)
    alt_input: bool = True
    # Should the reward reset after collecting a goal
    reward_reset: bool = False
    # Is the model still learning
    learning: bool = True
    # Path to file to dump metrics
    metrics_file: pathlib.Path = pathlib.Path(__file__).parent.parent / 'metrics.json'
    # Should the path cache be loaded
    load_path_cache: bool = False

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
    # Name of the shared memory that contains the agent structure
    player_name: str = 'agent'
    # Name of the shared memory that contains the arena structure
    arena_name: str = 'arena'

# Store variables related to the GUI implementation that should be shared between all GUI implementations
@dataclass(frozen=True)
class DefaultGUIDict:
    # Disable local GUI
    disable_gui: bool = False
    # The arena the GUI representation should be based from
    arena: Arena = None
    # Is the local gui running
    in_gui: bool = False
    # The start time of the current simulation
    start_time: float = 0.0
    # The end time of the current simulation state (only relevent when run_timer is False)
    end_time: float = 0.0
    # The current seed of the simulation
    seed: int = -1
    # Should the timer be running
    run_timer: bool = False
    # Start time for offset calculation
    offset_start_time: float = 0.0
    # Total offset time
    offset_time: float = 0.0
    # The current simulator time
    sim_time: float = 0.0
    # The texture backing the local GUI
    texture: np.ndarray = np.zeros(shape=(23 * 10, 23 * 10, 3), dtype=np.float32)
    # A copy of the previous state of the grid for delta generation
    previous_grid: np.ndarray = None
    # Size of each block in the texture
    block_size: int = 10

# Stores all variables related to the local and nengo gui representation
class GUIDict:
    def __init__(self) -> None:
        default_dict: dict[str] = asdict(DefaultGUIDict())
        default_dict = {'_' + k: v for k, v in default_dict.items()}
        self.__dict__.update(default_dict)
# Load all properties from the defaults for easy access
for attr, type in DefaultGUIDict.__annotations__.items():
        setattr(GUIDict, attr, create_property(attr, type))