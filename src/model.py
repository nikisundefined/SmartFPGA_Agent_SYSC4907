#!/usr/bin/env python3

import math
import json
import logging
import sys
import nengo
import nengo.learning_rules
import nengo.neurons
import nengo.solvers
import numpy as np
from dataclasses import dataclass
from simulation import Arena, Direction, Point, Player

@dataclass
class AttrDict:
    last_action: Direction = None # The last action performed by the agent in the simulation
    player_moved: bool = True # Did the player's location change from the last action
    ensemble_neurons: int = 400 # The number of neurons per ensemble
    learning_rate: float = 5e-6 # Learning rate of the learning rule
    neuron_type: nengo.neurons.NeuronType = nengo.SpikingRectifiedLinear() # Neuron type used in all ensembles
    solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=True) # Solver type used for learning connections
    learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate, pre_synapse=None) # Learning rule used for learning connections
    input_dimensions: int = 4 # Number of dimensions input to the model
    output_dimensions: int = 4 # Number of dimensions output from the model
    error_dimensions: int = 4 # Number of dimensions ouput from the error function
    dtype: np.dtype = np.float64 # The datatype used for all numpy arrays

    expected_reward: float = 0.0 # Last states expected reward
    reward: float = 1.0 # The current reward of the agent

    movement_threshold: float = 1e-6 # The minimum value for an action to be selected
    movement_weight: float = 0.8 # Weight of movement on error
    direction_weight: float = 0.2 # Weight of detection distance on error

    arena: Arena = Arena() # The arena the agent with move within
    action_performed: bool = False # Was an action performed by the agent since the last action
    in_gui: bool = False # In the simulation running in the nengo gui
    
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
    cvar.arena.move(index) # Move the player in the arena
    cvar.last_action = None if tmp == cvar.arena.player else (tmp - cvar.arena.player.point).asdirection() # Store the direction moved or 

    # Check if the player has stopped moving and log it
    if cvar.player_moved and tmp == cvar.arena.player:
        log.info(f"  Player has stopped moving at {cvar.arena.player}")
    else:
        cvar.arena.player.positions.append(tmp)
    cvar.player_moved = tmp != cvar.arena.player

    log.debug(f"  Direction {Direction(index).name} {x}")
    log.info(f"  Player Location: {cvar.arena.player} | Goal Location: {cvar.arena.goal}")

    # Update the goal location when the agent reaches the goal
    if cvar.arena.on_goal():
        log.info("  Agent reached the goal")
        cvar.arena.set_goal()
        # if np.any(cvar.reward < 0):
        #     cvar.reward = 0
    cvar.action_performed = True

    if cvar.in_gui:
        from PIL import Image
        arr = cvar.arena.display()
        if arr is None:
            return
        img = Image.fromarray(arr)
        img.save("arena.png")

### End Output Node Functions ###

### Error Function ###

# Calculates the error of the model inputs to outputs based on:
# NOTE: Differences of points (e.g. Goal and Player) are given as the absolute distances in the 4 directions North, East, South, West (NESW)
#   - The last location of the agent (LL)
#   - The distance from the agent to the goal (GL)
#   - The detection distance in all directions (SD)
def error(t: float, x: np.ndarray, cvar: AttrDict = cvar) -> np.ndarray:
    BASELINE_ERROR: float = 0.8 # The maximum value for the error

    # Get the best path to the goal
    path: list[Point] = cvar.arena.distance() # TODO: Check if this is reduced to a lookup in between movements
    # Compute the best direction based on the best path
    best_direction: Direction = (cvar.arena.player.point - path[1]).asdirection()

    # Compute the error for an early return
    # Error is the baseline value scaled by the inverse of the reward in the best direction
    err: np.ndarray = np.zeros(4) # Set the error to the negative of the post to reset all values
    err_scale: float = (1 / cvar.reward) ** 5 # Factor to scale error by
    err_val: float = x[best_direction] - BASELINE_ERROR * 1 # Compute the error value for the best direction
    # Set the direction we want to go in to the computed error value
    err[best_direction] = err_val
    # Prevent the error from going beyond the baseline error value
    err = err.clip(-BASELINE_ERROR, BASELINE_ERROR)

    # Only update the error when the agent has performed an action
    if not cvar.action_performed:
        return err
    cvar.action_performed = False

    # Compute the reward value for the current timestep
    def reward(player: Player, player_moved: bool, path: list[Point]) -> float:
        MOVEMENT_REWARD: float = 1.0 # The rewarda for moving (doubles as the penalty for not moving)

        # The number of steps to the goal (minus the current [0] and goal [len - 1] locations)
        dist_to_goal: int = len(path) - 2
        # How many steps did the agent needd to get to the goal in the last step
        previous_distance: int = len(cvar.arena.distance(start=player.positions[-1]))
        # Did the agent move towards the goal in this timestep
        moved_towards_goal: bool = dist_to_goal < previous_distance
        # Did the agent hit a wall in this timestep
        hit_wall: bool = player.positions[-1] == player.point

        reward: float = 0.0
        # Reward the agent for moving and punish for not moving
        reward += MOVEMENT_REWARD if player_moved else -MOVEMENT_REWARD
        # Reward the agent for moving towards the goal, but do not punish if it did not
        if moved_towards_goal:
            reward += 2 * MOVEMENT_REWARD
        # Punish the agent for moving/hitting a wall
        if hit_wall:
            reward -= 2 * MOVEMENT_REWARD
        return reward

    # Adjust the current state based on the current reward
    cvar.reward += reward(cvar.arena.player, cvar.player_moved, path)
    # Ensure that the minimum reward state possible is 1 for the later division to always succeed
    cvar.reward = max(cvar.reward, 1)

    # Recompute the error with the updated reward
    err: np.ndarray = np.zeros(4)
    err_scale: float = (1 / cvar.reward) ** 5
    err_val: float = x[best_direction] - BASELINE_ERROR * 1
    err[best_direction] = err_val
    err = err.clip(-1, 1)
    log.debug(f'  Updated error to: {err}')
    log.debug(f'  Updated reward to: {cvar.reward}')
    
    return err

def error_new(t: float, x: np.ndarray, cvar: AttrDict = cvar):
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
    #d_f: float = 0.9 # Discount factor

    def reward(player: Point, goal: Point, previous_position: Point, cvar: AttrDict = cvar):
        GOAL_W: float = 0.6 # Weight of the goal
        WALL_W: float = 0.3 # Weight of hitting the wall
        WALL_PENALTY: float = 1.0 # The penalty for hitting a wall
        # The minimum number of steps to the goal
        dis_to_goal: int = len(cvar.arena.distance())
        # The previous distance to the goal
        previous_distance: int = len(cvar.arena.distance(start=previous_position))
        # Did the agent move towards the goal (absolute distance)
        moved_towards_goal: bool = dis_to_goal < previous_distance
        # Did the agent hit the wall (last action caused the agent to not move and move into a wall)
        hit_wall: bool = player == previous_position

        #print(f'Moved towards goal: {moved_towards_goal} [{dis_to_goal} < {previous_distance}] {player} {goal} {previous_position} | {cvar.arena.player.positions}')

        reward: float = 0
        if moved_towards_goal:
            reward += GOAL_W * (previous_distance - dis_to_goal)
        elif cvar.player_moved:
            reward -= GOAL_W * (dis_to_goal - previous_distance)
        if hit_wall and cvar.action_performed:
            reward -= WALL_W * WALL_PENALTY
        if len(set(cvar.arena.player.positions[:-5])) < 3 and len(cvar.arena.player.positions) > 5:
            reward -= WALL_W * WALL_PENALTY
        return reward
    
    path: list[Point] = cvar.arena.distance() # Compute the shortest path to the goal
    #expected_reward_ns: float = reward(path[0], cvar.arena.goal, cvar.arena.player.point)
    # Error is given as the difference of reward values
    if math.isclose(t,int(t)):
        reward_value: float = reward(
                cvar.arena.player.point, 
                cvar.arena.goal, 
                cvar.arena.player.positions[-1] if len(cvar.arena.player.positions) > 0 else cvar.arena.player.point
            )
        # error: np.ndarray = np.array(
        #     [reward(cvar.arena.player.point + Direction.UP.topoint(), cvar.arena.goal, cvar.arena.player.point), 
        #     reward(cvar.arena.player.point + Direction.RIGHT.topoint(), cvar.arena.goal, cvar.arena.player.point),
        #     reward(cvar.arena.player.point + Direction.DOWN.topoint(), cvar.arena.goal, cvar.arena.player.point),
        #     reward(cvar.arena.player.point + Direction.LEFT.topoint(), cvar.arena.goal, cvar.arena.player.point)]
        # )
        error: np.ndarray = np.zeros(4)
        error[(cvar.arena.player.point - path[1]).asmagnitude().argmax()] = -reward_value
        error = error + 0.1 * cvar.reward
        cvar.reward = error
        print(f"Reward Value: {error}")
    else:
        error = cvar.reward
    # error: np.ndarray = np.ones(4) * reward_value - x[:4] * np.clip(x[4:], a_min=0, a_max=4/23)
    # player_last_pos: Point = cvar.arena.player.positions[-1] if len(cvar.arena.player.positions) > 0 else cvar.arena.player.point
    
    # if path is not None and len(path) > 2:
    #     error = error - (cvar.arena.player - path[0]).asmagnitude(cvar.dtype) * 0.01
    #cvar.expected_reward = expected_reward_ns # Store the expected reward for the next step
    
    # error = error.clip(-1, 1)
    return error

# Get the distance to a wall in every direction starting from the agent
def detection(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    return cvar.arena.detection().astype(cvar.dtype)

# Convert the current state of the arena into an RGBA pixel array
def generate_grid_image(arena: Arena, block_size: int = 10) -> np.ndarray:
    if not hasattr(generate_grid_image, 'shape'):
        generate_grid_image.__setattr__('shape', (3, arena.n * block_size, arena.m * block_size))
    #TODO: return numpy 3DD array of image of arena grid
    texture_data: np.ndarray = np.zeros(generate_grid_image.shape)
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

def limiter(t: float, x: np.ndarray):
    if np.any(x > 1):
        return 1
    elif np.any(x < -1):
        return -1
    return 0

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
    # Movement output
    mov_out = nengo.Node(
        output=move,
        size_in=cvar.output_dimensions,
        label='Movement Output'
    )
    # Error computation Input/Output
    err_tra = nengo.Node(
        output=error,
        size_in=cvar.error_dimensions,
        size_out=cvar.output_dimensions,
        label='Error Compute',
    )
    # shutoff = nengo.Node(
    #     output=limiter,
    #     size_in=cvar.output_dimensions,
    #     label='Shutoff'
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
        dimensions=cvar.output_dimensions,
        neuron_type=cvar.neuron_type,
        label='Error',
    )

    # Processing Connections
    conn_dist_in = nengo.Connection(
        pre=dist_in,
        post=pre,
        function=lambda x: x / 15,
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
    conn_err_tra = nengo.Connection(
        pre=err_tra,
        post=err,
        label='Error Transformation Connection'
    )
    conn_post_err = nengo.Connection(
        pre=post,
        post=err_tra,
        label='Post Feedback'
    )
    # conn_pre_err = nengo.Connection(
    #     pre=pre,
    #     post=err_tra[4:],
    #     label='Pre Feedback'
    # )
    conn_learn = nengo.Connection(
        pre=err,
        post=conn_pre_post.learning_rule,
        function=lambda x: [0.8, 0.7, 0.6, 0.5],
        label='Learning Connection'
    )

    # Limiter
    # conn_post_limit = nengo.Connection(
    #     pre=post,
    #     post=shutoff,
    # )
    # conn_limit_post = nengo.Connection(
    #     pre=shutoff,
    #     post=post,
    #     transform=-10*np.ones((post.size_in,1))
    # )

# Main function that displays a GUI of the arena and agent and runs the simulator for the agent with one time step per frame upto target_frame_rate
def main():
    import gui
    import time
    import dearpygui.dearpygui as dpg
    import pause

    nengo.logger.setLevel('WARNING')

    target_frame_rate: int = 30
    target_frame_time: float = 1.0/target_frame_rate

    gui.create_gui(cvar.arena)
    with nengo.Simulator(model, dt=(1.0/target_frame_rate), progress_bar=False) as sim:
        dpg.add_text(f"{sim.seed}", tag='seed', parent='Pacman') # Add custom text box displaying simulator seed
        
        gui.display_gui() # Display GUI
        log.info(f'Starting simulator with seed: {sim.seed}')
        while dpg.is_dearpygui_running():
            start_time = time.time() # Capture the start of frame computation time
            sim.run_steps(int(1/sim.dt)) # Perform one frame of the simulaton
            gui.update_text() # Update text boxes in the gui
            dpg.set_item_pos('seed', [dpg.get_viewport_width()/2-dpg.get_item_rect_size('seed')[0]/2, 265]) # Update custom text box added earlier
            gui.update_grid(cvar.arena) # Update the arena representation inside the GUI
            dpg.render_dearpygui_frame() # Render the updated frame to the GUI
            expected_end_time: float = start_time + target_frame_time # Compute how long to wait for the next frame
            pause.until(expected_end_time) # Wait until its time to render the next frame
        log.debug(f'Simulator ran: {sim.n_steps} steps')
    dpg.destroy_context()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Allow changing the logging level by command line parameter
    if len(sys.argv) > 1:
        if '--gui' in sys.argv:
            import os
            import gui
            import dearpygui.dearpygui as dpg

            ARENA_FILE: str = "arena.png"
            if not os.path.exists(ARENA_FILE):
                print("Waiting for simulation to start")
                while not os.path.exists(ARENA_FILE):
                    pass
                print("Simulation started")
            
            gui.create_gui(cvar.arena)
            while dpg.is_dearpygui_running():
                time = os.path.getmtime(ARENA_FILE)
                while time == os.path.getmtime(ARENA_FILE):
                    pass
                width, height, channels, data = dpg.load_image(ARENA_FILE)
                dpg.set_value('Environment', data)
                dpg.render_dearpygui_frame()
            if os.path.exists(ARENA_FILE):
                os.remove(ARENA_FILE)
            dpg.destroy_context()
        elif '--log' in sys.argv:
            if len(sys.argv) < 3:
                log.critical('Insufficient arguments for arg: --log')
                exit(2)
            if sys.argv.index('--log') == len(sys.argv) - 1:
                log.critical('Insufficient arguments for arg: --log')
            level = sys.argv[sys.argv.index('--log') + 1].upper()
            if level not in logging._nameToLevel:
                log.critical(f'Unknown logging level: {level}')
            log.setLevel(level)

    # Catch any errors gracefully and exit
    try:
        main()
    except Exception as e:
        log.critical(f"ERROR: {e}", exc_info=1)
        exit(1)

if '__page__' in locals():
    # cvar.in_gui = True # Indicate that the script is run in nengo gui
    log = nengo.logger
    log.setLevel('INFO')
    print('Setting up for NengoGUI')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)