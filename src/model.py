#!/usr/bin/env python3

import math
import json
import logging
import nengo
import nengo.learning_rules
import nengo.neurons
import nengo.solvers
import numpy as np
from dataclasses import dataclass
from lib.simulation import Arena, Direction, Point, Player

@dataclass
class AttrDict:
    last_action: Direction = None # The last action performed by the agent in the simulation
    player_moved: bool = True # Did the player's location change from the last action
    ensemble_neurons: int = 100 # The number of neurons per ensemble
    learning_rate: float = 5e-6 # Learning rate of the learning rule
    neuron_type: nengo.neurons.NeuronType = nengo.SpikingRectifiedLinear() # Neuron type used in all ensembles
    solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=True) # Solver type used for learning connections
    learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate) # Learning rule used for learning connections
    input_dimensions: int = 4 # Number of dimensions input to the model
    output_dimensions: int = 4 # Number of dimensions output from the model
    error_dimensions: int = 4 # Number of dimensions ouput from the error function
    dtype: np.dtype = np.float64 # The datatype used for all numpy arrays

    expected_reward: float = 0.0 # Last states expected reward

    movement_threshold: float = 1e-9 # The minimum value for an action to be selected
    movement_weight: float = 0.8 # Weight of movement on error
    direction_weight: float = 0.2 # Weight of detection distance on error

    arena: Arena = Arena() # The arena the agent with move within
    action_performed: bool = False # Was an action performed by the agent since the last action
    
    def export(self) -> str:
        class JsonEncoder(json.JSONEncoder):
            def __init__():
                super()
            
            def default(self, o):
                if o is Arena or o is Player or o is Point:
                    return o.__json__()
                elif o is Direction:
                    return int(o)
                return super().default(o)
        return json.dumps(self.__dict__, cls=JsonEncoder, skipkeys=True)

# Setup the variables for the model
cvar: AttrDict = AttrDict()
log = logging.getLogger('simulation')

### Input Node Functions ###

# Returns the current location of the player as a 2D Point
def player_location(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    return np.array([cvar.arena.player.x, cvar.arena.player.y], dtype=cvar.dtype)

# Returns the current location of the goal as a 2D Point
def goal_location(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    return np.array([cvar.arena.goal.x, cvar.arena.goal.y], dtype=cvar.dtype)

# Return the last location of the player
def last_location(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    if len(cvar.arena.player.positions) == 0:
        return player_location(t)
    return np.array([cvar.arena.player.positions[-1].x, cvar.arena.player.positions[-1].y], dtype=cvar.dtype)

# Returns the distance between the current and last location of the player as a 4D Distance
def delta_distance(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    tmp = player_location(t) - last_location(t)
    tmp = np.array([-tmp[1], tmp[0], tmp[1], -tmp[0]], dtype=cvar.dtype)
    return np.maximum(tmp, np.zeros(4, dtype=cvar.dtype))

# Return the distance between the player and the goal as a 4D Distance
def goal_distance(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    tmp: Point = cvar.arena.player - cvar.arena.goal
    return tmp.asmagnitude(dtype=cvar.dtype)

### End Input Node Functions ###

### Output Node Functions ###

# Moves the agent in the arena based on the index of the largest value provided
# Only moves the agent every 1 second
def move(t: float, x: np.ndarray, cvar: AttrDict = cvar):
    if not math.isclose(t, int(t)):
        return
    log.info(f"Move at {round(t, 2)}")

    # Determine the action to perform (Direction to move)
    index = int(np.argmax(x))
    if math.isclose(x[index],0,abs_tol=cvar.movement_threshold): # Check if the input value was larger than the threshold
        log.info(f"  No action selected")
        cvar.player_moved = False
        return

    # Ensure the index is in range of the enum to prevent errors (TODO: unneeded?)
    if index < 0 or index > 3:
        raise IndexError(f'simulation.Direction: Index {index} out of range')
    tmp: Point = Point(cvar.arena.player.x, cvar.arena.player.y) # Store the old location
    cvar.arena.move(index)

    # Check if the player has stopped moving and log it
    if cvar.player_moved and tmp == cvar.arena.player:
        log.info(f"  Player has stopped moving at {cvar.arena.player}")
    cvar.player_moved = tmp != cvar.arena.player

    log.debug(f"  Direction {Direction(index).name} {x}")
    log.info(f"  Player Location: {cvar.arena.player} | Goal Location: {cvar.arena.goal}")

    # Update the goal location when the agent reaches the goal
    if cvar.arena.on_goal():
        log.info("  Agent reached the goal")
        cvar.arena.set_goal()
    cvar.action_performed = True

### End Output Node Functions ###

### Error Function ###

# Calculates the error of the model inputs to outputs based on:
# NOTE: Differences of points (e.g. Goal and Player) are given as the absolute distances in the 4 directions North, East, South, West (NESW)
#   - The last location of the agent (LL)
#   - The distance from the agent to the goal (GL)
#   - The detection distance in all directions (SD)
def error(t: float, x: np.ndarray, cvar: AttrDict = cvar):
    WEIGHT_ADJUSTMENT: float = 0.01 # Amount to adjust error weights by

    # Extract the inputs from the compressed input
    ll: np.ndarray = x[0:4]
    gl: np.ndarray = x[4:8]
    sd: np.ndarray = x[8:12]

    # Only update the error weights every ~1 second
    if math.isclose(t,int(t)):
        return gl + sd

    # Check if the agent is moving in the direction that has the most open space
    if cvar.last_action and np.argmax(sd) != int(cvar.last_action):
        cvar.direction_weight += WEIGHT_ADJUSTMENT
        cvar.movement_weight -= WEIGHT_ADJUSTMENT
    else:    
        sd *= cvar.direction_weight
    
    # Check if the agent moved during the last action
    if not ll.any(0):
        cvar.movement_weight += WEIGHT_ADJUSTMENT
        cvar.direction_weight -= WEIGHT_ADJUSTMENT
        gl *= cvar.movement_weight

    # Compute and return error
    err = gl + sd
    return err

def error_new(t: float, cvar: AttrDict = cvar):
    """
dis_to_goal = root((Xc-Xg)^2 + (Yx-Yg)^2)
reward(t) = + goal_w * (previous_distance - dis_to_goal) if moved towards
        - wall_w * wall_penalty              if hits wall
        - goal_w * (dis_to_goal - previous_distance) if moves away 
        0  if no change

goal_w = scaling factor
wall_w = scaling factor
wall_penalty = binary variable (1/0)

error calc:

error = reward(t) + d_f * expected_reward_ns - expected_reward

learning rate lr = 
weight_update = lr * error * si(t) * sj(t)

si = pre
sj = post
"""
    d_f: float = 0.9 # Discount factor

    def reward(player: Point, goal: Point, previous_position: Point, cvar: AttrDict = cvar):
        GOAL_W: float = 0.7 # Weight of the goal
        WALL_W: float = 0.6 # Weight of hitting the wall
        WALL_PENALTY: float = 1.0 # The penalty for hitting a wall
        # The current absolute distance to the goal
        dis_to_goal: float = player.distance(goal)
        # The previous distance to the goal
        previous_distance: float = previous_position.distance(goal)
        # Did the agent move towards the goal (absolute distance)
        moved_towards_goal: bool = dis_to_goal < previous_distance
        # Did the agent hit the wall (last action caused the agent to not move and move into a wall)
        hit_wall: bool = player == previous_position # TODO: Validate this equation

        reward: float = 0
        if moved_towards_goal:
            reward += GOAL_W * (previous_distance - dis_to_goal)
        else:
            reward -= GOAL_W * (dis_to_goal - previous_distance)
        if hit_wall:
            reward -= WALL_W * WALL_PENALTY
        return reward
    
    # Error is given as the difference of reward values
    error: float = reward(cvar.arena.player.point, cvar.arena.goal, cvar.arena.player.positions[-1] if len(cvar.arena.player.positions) > 0 else cvar.arena.player.point) + d_f * (cvar.expected_reward)

    # Compute the expected reward of the next step
    # The expected reward is based on if the agent was to to choose the best direction
    goal_direction: np.ndarray = (cvar.arena.player - cvar.arena.goal).asmagnitude(cvar.dtype) # Compute which directions the goal is in
    goal_distances: np.ndarray = cvar.arena.detection() * goal_direction # Isolate the distances from the detection
    if (goal_distances == 0).all():
        goal_distances = cvar.arena.detection()
    best_direction: np.ndarray = np.zeros(4) # Find the best direction as a vector
    best_direction[np.argmax(goal_distances)] = 1
    best_direction = Point.fromvector(best_direction) # Convert the vector into a point for further math
    cvar.expected_reward = reward(cvar.arena.player + best_direction, cvar.arena.goal, cvar.arena.player.point)
    return error

# Get the distance to a wall in every direction starting from the agent
def detection(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    return cvar.arena.detection().astype(cvar.dtype)

# Returns 1 when an action has been performed (used to reset the post ensemble after taking an action)
def limiter(t: float, cvar: AttrDict = cvar):
    if cvar.action_performed:
        cvar.action_performed = False
        return 1
    return 0

# Returns 1 if any value in x is greater than 1 (used to prevent the value of post from becoming too large)
def shutoff(t: float, x) -> np.ndarray:
    for val in x:
        if val > 1:
            return 1
    return 0

# Convert the current state of the arena into an RGBA pixel array
def generate_grid_image(arena: Arena, block_size: int = 10):
    texture_data: list[float] = []
    _map: dict[int, list[float]] = {
        Arena.EMPTY: [0, 0, 0, 0], # Black
        Arena.WALL: [0, 0, 255 / 255, 255 / 255], # Blue
        Arena.PLAYER: [255/ 255, 255 / 255, 0, 255/ 255], # Yellow
        Arena.GOAL: [0, 255 / 255, 0, 255 / 255], # Green
    }

    for y in range(arena.m): # Every Y coordinate
        for _ in range(block_size): # Every Y block
            for x in range(arena.n): # Every X coordinate
                for _ in range(block_size): # Every X block
                    # RGBA pixel format
                    texture_data.extend(_map[arena.grid[y][x]]) # Pixel value
    return texture_data

# Global model definition for use with NengoGUI
model = nengo.Network(label='pacman')
with model:
    bg = nengo.networks.BasalGanglia(dimensions=cvar.output_dimensions)
    thal = nengo.networks.Thalamus(dimensions=cvar.output_dimensions)

    # Nodes (interaction with simulation)
    # Detection distance input
    dist_in = nengo.Node(
        output=detection,
        size_out=cvar.input_dimensions,
        label='Distance Input Node'
    )
    # Last location distance input
    # cur_in = nengo.Node(
    #     output=delta_distance,
    #     size_out=cvar.error_dimensions,
    #     label='Movement Change Input',
    # )
    # Goal distance input
    # gol_in = nengo.Node(
    #     output=goal_distance,
    #     size_out=cvar.error_dimensions,
    #     label='Goal Distance Input'
    # )
    # Movement output
    mov_out = nengo.Node(
        output=move,
        size_in=cvar.output_dimensions,
        label='Movement Output'
    )
    # Error computation Input/Output
    err_tra = nengo.Node(
        output=error_new,
        # size_in=cvar.error_dimensions*3,
        size_out=cvar.output_dimensions,
        label='Error Compute',
    )
    # Post reset
    # shutoff_gate = nengo.Node(
    #     output=limiter,
    #     label='Shutoff Gate'
    # )
    # # Post reset
    # flush = nengo.Node(
    #     output=shutoff,
    #     size_in=cvar.output_dimensions,
    #     size_out=cvar.output_dimensions,
    #     label='Post Flush'
    # )

    # Ensembles
    pre = nengo.Ensemble(
        n_neurons=cvar.ensemble_neurons,
        dimensions=cvar.input_dimensions,
        neuron_type=cvar.neuron_type,
        label='Pre',
    )
    post = nengo.Ensemble(
        n_neurons=cvar.ensemble_neurons,
        dimensions=cvar.output_dimensions,
        neuron_type=cvar.neuron_type,
        label='Post',
    )
    err = nengo.Ensemble(
        n_neurons=cvar.ensemble_neurons,
        dimensions=cvar.error_dimensions,
        neuron_type=cvar.neuron_type,
        label='Error',
    )

    # Processing Connections
    conn_dist_in = nengo.Connection(
        pre=dist_in,
        post=pre,
        function=lambda x: x / 23,
        label='Distance Input Connection',
    )
    conn_pre_post = nengo.Connection(
        pre=pre,
        post=post,
        learning_rule_type=cvar.learning_rule_type,
        label='Pre -> Post Connection',
    )

    # Output Filtering Connections
    conn_post_bg = nengo.Connection(
        pre=post,
        post=bg.input,
        label='Post -> BG Connection'
    )
    conn_bg_thal = nengo.Connection(
        pre=bg.output,
        post=thal.input,
        label='BG -> Thal Connection'
    )
    conn_thal_out = nengo.Connection(
        pre=thal.output,
        post=mov_out,
        label='Action Output Connection'
    )

    # Learning Connections
    # conn_play_loc = nengo.Connection(
    #     pre=cur_in,
    #     post=err_tra[0:4],
    #     label='Location Delta -> Error Connection'
    # )
    # conn_goal_loc = nengo.Connection(
    #     pre=gol_in,
    #     post=err_tra[4:8],
    #     transform=-1,
    #     label='Goal Distance -> Error Connection'
    # )
    # conn_dist_err = nengo.Connection(
    #     pre=dist_in,
    #     post=err_tra[8:12],
    #     label='Detection Distance -> Error Connection'
    # )
    conn_err_tra = nengo.Connection(
        pre=err_tra,
        post=err,
        label='Error Transformation Connection'
    )
    conn_learn = nengo.Connection(
        pre=err,
        post=conn_pre_post.learning_rule,
        label='Learning Connection'
    )

    # Reset Connections
    # conn_shutoff = nengo.Connection(
    #     pre=shutoff_gate, 
    #     post=post.neurons, 
    #     transform=-1000*np.ones((post.n_neurons, 1)), 
    #     synapse=None,
    #     label='Limiter'
    # )
    # conn_flush_post = nengo.Connection(
    #     pre=flush,
    #     post=post.neurons,
    #     transform=-1000000000*np.ones((post.n_neurons, 4)),
    #     label='Flush -> Post',
    # )
    # conn_post_flush = nengo.Connection(
    #     pre=post,
    #     post=flush,
    #     label='Post -> Flush',
    # )

# Main function that displays a GUI of the arena and agent and runs the simulator for the agent with one time step per frame upto target_frame_rate
def main():
    import lib.gui as gui
    import time
    import dearpygui.dearpygui as dpg
    import pause

    target_frame_rate: int = 30
    target_frame_time: float = 1.0/target_frame_rate

    gui.create_gui(cvar.arena)
    with nengo.Simulator(model, dt=(1.0/target_frame_rate)) as sim:
        dpg.add_text(f"{sim.seed}", tag='seed', parent='Pacman') # Add custom text box displaying simulator seed
        
        gui.display_gui() # Display GUI
        while dpg.is_dearpygui_running():
            start_time = time.time() # Capture the start of frame computation time
            sim.step() # Perform one step of the simulaton
            gui.update_text() # Update text boxes in the gui
            dpg.set_item_pos('seed', [dpg.get_viewport_width()/2-dpg.get_item_rect_size('seed')[0]/2, 265]) # Update custom text box added earlier
            gui.update_grid(cvar.arena) # Update the arena representation inside the GUI
            dpg.render_dearpygui_frame() # Render the updated frame to the GUI
            expected_end_time: float = start_time + target_frame_time # Compute how long to wait for the next frame
            pause.until(expected_end_time) # Wait until its time to render the next frame
        log.debug(f'Simulator ran: {sim.n_steps} steps')
    dpg.destroy_context()

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    # Allow changing the logging level by command line parameter
    if len(sys.argv) > 1:
        if '--log' in sys.argv:
            if len(sys.argv) < 3:
                logging.critical('Insufficient arguments for arg: --log')
                exit(2)
            if sys.argv.index('--log') == len(sys.argv) - 1:
                logging.critical('Insufficient arguments for arg: --log')
            level = sys.argv[sys.argv.index('--log') + 1].upper()
            if level not in logging._nameToLevel:
                logging.critical(f'Unknown logging level: {level}')
            log.setLevel(level)

    print(error_new(0))

    # Catch any errors gracefully and exit
    try:
        main()
    except Exception as e:
        log.critical(f"ERROR: {e}", exc_info=1)
        exit(1)