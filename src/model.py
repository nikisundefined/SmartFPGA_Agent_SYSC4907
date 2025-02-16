#!/usr/bin/env python3

import math
import json
import logging
import sys
import nengo
import nengo.learning_rules
import nengo.neurons
import nengo.solvers
import nengo_gui
import nengo_fpga.networks
import numpy as np
import gui
import time
import dearpygui.dearpygui as dpg
import pause
import threading
from dataclasses import dataclass
from simulation import Arena, Direction, Point, Player

# NOTE: 
#   Consider Super/Sub reward state (Super state = Score * Time, Sub state = distance to goal)
#   Scale Super/Sub reward state error differently (Super state = scale global error, Sub state = scale directional error)
#   Current version resets reward on collection of goal
#   Consider increasing the error for the second best move in the path
#   Consider making a web based display to allow running the model truely headless

# TODO:
#   Add toggle for alternate input method
#       - Find better inputs
#   Check if the model is actually learning or just adapting based on the error
#       - Check this with performance characteristics
#   Update score in local gui
#   Make error function tend to be more punishing
#   Add more tracking in player class (Steps to reach goal, Reward value, Time taken)
#   Add performance characteristics:
#       - Time per goal
#       - Movements per goal
#       - Reward Value at goal
#   Validate best path can properly wrap
#   Change error scale function
#   Cleanup main script code
#   Generate list of hyperparameters for optimization phase
#       - Learning Rate
#       - Error Baseline

@dataclass
class AttrDict:
    # The last action performed by the agent in the simulation
    last_action: Direction = None
    # Did the player's location change from the last action
    player_moved: bool = True
    # The number of neurons per ensemble
    ensemble_neurons: int = 400
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
    # In the simulation running in the nengo gui
    in_gui: bool = False
    # Flag if the current action moved away from the goal
    moved_away_from_goal: int = 0
    # The synapse used for the connection between pre and post ensembles
    connection_synapse = 0.01
    # The logging level for the script to print out
    log_level: str | int = int(logging.DEBUG)
    
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

# Returns the distance to the goal as the length of the path calculated using the A* algorithm
def goal_path_distance(t: float, cvar: AttrDict = cvar) -> int:
    return len(cvar.arena.distance()) / 30.0

def goal_best_direction(t: float, cvar: AttrDict = cvar) -> float:
    return (int(cvar.arena.best_direction()) + 1) / 5.0

def goal_point_distance(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    delta: Point = cvar.arena.player.point - cvar.arena.goal
    return np.array([delta.x, delta.y], dtype=cvar.dtype) / 23.0

### End Input Node Functions ###

### Output Node Functions ###

# Moves the agent in the arena based on the index of the largest value provided
# Only moves the agent every 1 second
def move(t: float, x: np.ndarray, cvar: AttrDict = cvar):
    if t == 0.0 and x is None and cvar.in_gui:
        dpg.set_value('timer', time.time()) # Reset the timer if the model was reloaded
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
    delta_dist: Point = tmp - cvar.arena.player.point
    cvar.last_action = None if tmp == cvar.arena.player else (Point(np.sign(delta_dist.x), 0).asdirection() if abs(delta_dist.x) == 22 else delta_dist.asdirection()) # Store the direction moved or 

    # Check if the player has stopped moving and log it
    if cvar.player_moved and tmp == cvar.arena.player:
        log.info(f"  Player has stopped moving at {cvar.arena.player}")
    else:
        cvar.arena.player.positions.append(tmp)
    cvar.player_moved = tmp != cvar.arena.player

    log.debug(f"  Direction {Direction(index).name} {x}")
    log.info(f"  Player Location: {cvar.arena.player} -> {Direction(index).name} | Goal Location: {cvar.arena.goal}")

    # Update the goal location when the agent reaches the goal
    if cvar.arena.on_goal():
        log.info("  Agent reached the goal")
        log.debug(f"  Player score is now: {cvar.arena.player.score}")
        cvar.arena.set_goal()
    cvar.action_performed = True

    if cvar.in_gui:
        gui.update_grid(cvar.arena)

### End Output Node Functions ###

### Error Function ###

# Calculates the error of the model inputs to outputs based on:
def error(t: float, x: np.ndarray, cvar: AttrDict = cvar) -> np.ndarray:
    BASELINE_ERROR: float = 0.8 # The maximum value for the error
    ERROR_CURVE: float = 2.0
    
    def err_calc(best_direction: Direction, x: np.ndarray = x, cvar: AttrDict = cvar):
        # Error is the baseline value scaled by the inverse of the reward in the best direction
        err: np.ndarray = x # Set the error to the post to reset all values other than the best direction
        err_scale: float = (ERROR_CURVE / cvar.reward) # Factor to scale error by
        err_val: float = (x[best_direction] - BASELINE_ERROR) * err_scale  # Compute the error value for the best direction
        # Set the direction we want to go in to the computed error value
        err[best_direction] = err_val
        # Prevent the error from going beyond the baseline error value
        err = err.clip(-BASELINE_ERROR, BASELINE_ERROR)
        return err

    # Get the best path to the goal
    path: list[Point] = cvar.arena.distance()
    # Compute the best direction based on the best path
    delta_dist: Point = path[1] - cvar.arena.player.point
    best_direction: Direction = Point(np.sign(delta_dist.x), 0).asdirection() if abs(delta_dist.x) == 22 else delta_dist.asdirection()

    # Compute the error for an early return
    err = err_calc(best_direction)

    # Only update the error when the agent has performed an action
    if not cvar.action_performed:
        return err
    cvar.action_performed = False
    log.debug(f'  Best Direction is: {best_direction.name}')
    log.debug(f'  Best Path: {path[1:-1]}')

    # Compute the reward value for the current timestep
    def reward(player: Player, player_moved: bool, path: list[Point]) -> float:
        MOVEMENT_REWARD: float = 1.0 # The rewarda for moving (doubles as the penalty for not moving)

        # The number of steps to the goal (minus the current [0] and goal [len - 1] locations)
        dist_to_goal: int = len(path) - 2
        # How many steps did the agent needd to get to the goal in the last step
        previous_distance: int = len(cvar.arena.distance(start=player.positions[-1])) - 2
        # Did the agent move towards the goal in this timestep
        moved_towards_goal: bool = dist_to_goal < previous_distance
        # Did the agent hit a wall in this timestep
        hit_wall: bool = player.positions[-1] == player.point

        reward: float = 0.0
        # Reward the agent for moving and punish for not moving
        reward += MOVEMENT_REWARD if player_moved else -MOVEMENT_REWARD
        # Reward the agent for moving towards the goal
        if moved_towards_goal:
            reward += 2 * MOVEMENT_REWARD
            # Reset the counter for moving away from the goal
            cvar.moved_away_from_goal = 0
        else:
            # Increment the counter
            cvar.moved_away_from_goal += 1
            # If the last 2 movements mvoed away from the goal
            if cvar.moved_away_from_goal == 2:
                # Punish it more
                reward -= 3 * MOVEMENT_REWARD
                cvar.moved_away_from_goal = 0
            else:
                reward -= MOVEMENT_REWARD
        # Punish the agent for moving/hitting a wall
        if hit_wall:
            reward -= 2 * MOVEMENT_REWARD
        # Punish the agent for returning to the same points
        if len(player.positions) > 5:
            unique_positions: int = len(set(player.positions[-5:]))
            log.debug(f"  Unique Positions: {unique_positions}")
            if unique_positions <= 2:
                reward -= 3 * MOVEMENT_REWARD
            elif unique_positions <= 4:
                reward -= MOVEMENT_REWARD
        log.debug(f"  Instantaneous Reward: {reward}")
        return reward

    # Adjust the current state based on the current reward
    cvar.reward += reward(cvar.arena.player, cvar.player_moved, path)
    # Ensure that the minimum reward state possible is 1 for the later division to always succeed
    cvar.reward = max(cvar.reward, 1)

    # Recompute the error with the updated reward
    err = err_calc(best_direction)
    log.debug(f'  Updated error to: {err}')
    log.debug(f'  Updated reward to: {cvar.reward}')
    
    return err

# Get the distance to a wall in every direction starting from the agent
def detection(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    # Get the detection information from the arena
    tmp = cvar.arena.detection().astype(cvar.dtype)
    # Convert the detection distance to a binary value base on if there is or is not a wall in a direction
    return tmp.clip(0, 1)

def create_model_fpga():
    global model
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
        g_dist = nengo.Node(
            output=goal_path_distance,
            size_out=1,
            label='Goal Path Distance'
        )
        best_dir = nengo.Node(
            output=goal_best_direction,
            size_out=1,
            label='Goal Best Direction'
        )
        g_pnt = nengo.Node(
            output=goal_point_distance,
            size_out=2,
            label='Goal Point Distance'
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
        nreward = nengo.Node(
            output=lambda x: cvar.reward,
            size_out=1,
            label='Reward'
        )

        # Ensembles
        fpga = nengo_fpga.networks.FpgaPesEnsembleNetwork(
            "ADDME",
            n_neurons=cvar.ensemble_neurons,
            dimensions=cvar.output_dimensions,
            learning_rate=cvar.learning_rate,
            label='FPGA'
        )
        if cvar.neuron_type in [nengo.neurons.SpikingRectifiedLinear, nengo.neurons.RectifiedLinear]:
            fpga.ensemble.neuron_type = cvar.neuron_type
        fpga.connection.solver = cvar.solver_type
        fpga.connection.synapse = cvar.connection_synapse
        err = nengo.Ensemble(
            n_neurons=cvar.ensemble_neurons,
            dimensions=cvar.output_dimensions,
            neuron_type=cvar.neuron_type,
            label='Error',
        )

        # Processing Connections
        conn_dist_in = nengo.Connection(
            pre=dist_in,
            post=fpga.input,
            label='Distance Input Connection',
        )
        # conn_g_dist_in = nengo.Connection(
        #     pre=g_dist,
        #     post=fpga.input[0],
        #     label='Goal Distance Input'
        # )
        # conn_best_dir_in = nengo.Connection(
        #     pre=best_dir,
        #     post=fpga.input[1],
        #     label='Best Direction Input'
        # )
        # conn_pnt_dist_in = nengo.Connection(
        #     pre=g_pnt,
        #     post=fpga.input[2:],
        #     label='Goal Point Distance Input'
        # )


        # Output Filtering Connections
        conn_post_bg = nengo.Connection(
            pre=fpga.output,
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
            pre=bg.input,
            post=err_tra,
            label='Post Feedback'
        )
        conn_learn = nengo.Connection(
            pre=err,
            post=fpga.error,
            label='Learning Connection'
        )


def create_model():
    global model
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
        nreward = nengo.Node(
            output=lambda x: cvar.reward,
            size_out=1,
            label='Reward'
        )

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
            label='Distance Input Connection',
        )
        # conn_inp = nengo.Connection()
        conn_pre_post = nengo.Connection(
            pre=pre,
            post=post,
            synapse=cvar.connection_synapse,
            learning_rule_type=cvar.learning_rule_type,
            solver=cvar.solver_type,
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
        conn_learn = nengo.Connection(
            pre=err,
            post=conn_pre_post.learning_rule,
            label='Learning Connection'
        )

# Main function that displays a GUI of the arena and agent and runs the simulator for the agent with one time step per frame upto target_frame_rate
def main():
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

def web_gui():
    gui.create_gui(cvar.arena)
    
    gui.display_gui() # Display GUI
    while dpg.is_dearpygui_running():
        jobs = dpg.get_callback_queue()
        dpg.run_callbacks(jobs)

        gui.update_text() # Update text boxes in the gui
        dpg.set_value('score', f'Score: {cvar.arena.player.score}')

        dpg.render_dearpygui_frame() # Render the updated frame to the GUI
        time.sleep(0.1)
    dpg.destroy_context()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Allow changing the logging level by command line parameter
    if len(sys.argv) > 1:
        if '--nengo' in sys.argv:
            # Start the gui in another thread
            t = threading.Thread(target=web_gui, name='GUI', daemon=True)
            t.start()
            # Start the nengo web gui in the main thread
            g = nengo_gui.GUI(filename=__file__, editor=True)
            g.start()
            # Ensure the script exits after this runs
            exit(0)
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
    # Check if the local gui is running and set flag
    for t in threading.enumerate():
        if t.name == 'GUI':
            log.info('Found gui')
            cvar.in_gui = True
            break
    
    # Setup logging
    log = nengo.logger
    log.setLevel(cvar.log_level)
    print('Setting up for NengoGUI')
    logging.basicConfig(stream=sys.stdout, level=cvar.log_level)
    # Create the model
    global model
    model: nengo.Network = None
    create_model_fpga()