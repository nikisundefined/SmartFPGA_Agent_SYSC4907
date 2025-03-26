#!/usr/bin/env python3
import sys
import math
import time
import logging
import threading
import json
import webbrowser

log = logging.getLogger('smart_agent.model')

import smart_agent
import nengo
import nengo.learning_rules
import nengo.neurons
import nengo.solvers
import nengo_gui
import nengo_gui.guibackend
import nengo_gui.page
import numpy as np
try:
    import dearpygui.dearpygui as dpg
    import smart_agent.gui as gui
except ImportError:
    log.warning(f"Failed to load DearPyGUI, unable to load local GUI")
    smart_agent.gvar.disable_gui = True

import smart_agent.vars as vars
import smart_agent.shared as shared
import smart_agent.simulation as simulation

# Import some names from other files to avoid rewriting all names in this file
AttrDict = vars.ConsoleDict
Direction = simulation.Direction
Point = simulation.Point
Player = simulation.Player
Arena = simulation.Arena
PathCache = simulation.PathCache

# Learning Inhibit Vars
LEARN_STOP_TIME: float = 240.0
LEARN_MEASURE_TIME: float = LEARN_STOP_TIME * 1.25

# Setup the variables for the model
cvar = smart_agent.cvar
gvar = smart_agent.gvar
log.setLevel(cvar.log_level)

if not cvar.load_path_cache:
    log.warning('Not loading path cache entries')
# If the shared cache is not loaded and the cache file exists, attempt to load it
elif len(shared.SharedArena.shared_path_cache) == 0 and cvar.path_cache_file.exists():
    attached: bool = False
    try:
        # Check if the cache is already stored in shared memory
        tmp = shared.create_shared_memory(0, 'path_cache', shared.AttachFlag.ATTACH)
        log.info('Found path cache in shared memory')
        # Ensure it is the correct size
        if tmp.nbytes < shared.SharedPathCache.size:
            log.warning(f'Shared memory region for path cache is incorrect size: {tmp.nbytes} < {shared.SharedPathCache.size}')
            tmp.release()
            # Not the right size, remove it and reallocate
            elem: str | None = None
            for name, value in shared.shm_names.items():
                if value[0].name == 'path_cache':
                    value[0].close()
                    value[0].unlink()
                    elem = name
                    break
            if elem is None:
                raise RuntimeError(f'Failed to find path_cache in shared memory')
            del shared.shm_names[elem]
            # Raise error to execute code related to loading from file
            raise BufferError()
        attached = True
        keys = shared.create_shared_memory(0, 'path_cache_keys', shared.AttachFlag.ATTACH)
    except:
        log.info('Could not find shared memory region for path cache, loading from file')
        # Could not load from shared memory, load from file instead
        Arena.path_cache = PathCache.fromfile(cvar.path_cache_file.absolute())
        log.info(f'Loaded {len(Arena.path_cache)} paths from {cvar.path_cache_file.absolute()}')
        nbytes: int = Arena.path_cache.count() * shared.SharedPoint.size
        tmp = shared.create_shared_memory(nbytes, 'path_cache')
    
    # Prepare the SharedPathCache
    shared.SharedArena.shared_path_cache = shared.SharedPathCache(tmp)
    # Load differently based on if the path cache was already stored in memory
    if attached:
        shared.SharedArena.shared_path_cache.loadkeys(keys)
    else:
        shared.SharedArena.shared_path_cache.load(Arena.path_cache)
    log.info(f"Loaded {len(shared.SharedArena.shared_path_cache)} paths from cache")

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
    return len(cvar.arena.distance()) / 44.0

def goal_best_direction(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    tmp = np.array([0, 0, 0, 0], dtype=cvar.dtype)
    tmp[int(cvar.arena.best_direction())] = 0.4 #update this to 0.2?
    return tmp

def goal_point_distance(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    delta: Point = cvar.arena.player.point - cvar.arena.goal
    return np.array([delta.x, delta.y], dtype=cvar.dtype) / 23.0 # + 0.3?

### End Input Node Functions ###

### Output Node Functions ###

# Moves the agent in the arena based on the index of the largest value provided
# Only moves the agent every 1 second
def move(t: float, x: np.ndarray, cvar: AttrDict = cvar):
    GOAL_COLLECT_REWARD: float = 0.0

    # Only attempt to move when t is an integer, ex. 1, 2, 3, ...
    if not math.isclose(t, int(t)):
        return
    log.info(f"Move at {round(t, 2)}")
    #update here
    cvar.arena.player.info.update_time()
    # Determine the action to perform (Direction to move)
    index = int(np.argmax(x))
    if math.isclose(x[index],0,abs_tol=cvar.movement_threshold): # Check if the input value was larger than the threshold
        log.info(f"No action selected")
        cvar.player_moved = False
        return

    tmp: Point = Point(cvar.arena.player.x, cvar.arena.player.y) # Store the old location
    cvar.arena.move(index) # Move the player in the arena
    delta_dist: Point = tmp - cvar.arena.player.point
    # Store the direction moved
    if tmp == cvar.arena.player:
        cvar.last_action == Direction.NONE # None if the player did not move
    elif abs(delta_dist.x) == cvar.arena.n - 1:
        cvar.arena.player.info.update_actions()
        cvar.last_action = Point(np.sign(delta_dist.x), 0).asdirection() # Special case to handle wrapping
    else:
        cvar.arena.player.info.update_actions()
        cvar.last_action = delta_dist.asdirection() # Generic form convert change in location to a direction

    # Check if the player has stopped moving and log it
    if cvar.player_moved and tmp == cvar.arena.player:
        log.info(f"Player has stopped moving at {cvar.arena.player}")
    else:
        cvar.arena.player.positions.append(tmp)
    cvar.player_moved = tmp != cvar.arena.player

    log.debug(f"Direction {Direction(index).name} {x}")
    log.info(f"Player Location: {cvar.arena.player} | Goal Location: {cvar.arena.goal}")

    # Update the goal location when the agent reaches the goal
    if cvar.arena.on_goal():
        log.info("Agent reached the goal")

        cvar.arena.set_goal()
        log.debug(f"Player score is now: {cvar.arena.player.score}")
        if cvar.reward_reset:
            cvar.reward = vars.DefaultConsoleDict.reward
        else:
            cvar.reward += GOAL_COLLECT_REWARD
        
        log.info(cvar.arena.performance)
    cvar.action_performed = True
    log.debug(f"Current detection: {detection(0)}")

    if gvar.in_gui:
        gui.update()

### End Output Node Functions ###

### Error Function ###

# Calculates the error of the model inputs to outputs based on:
def error(t: float, x: np.ndarray, cvar: AttrDict = cvar) -> np.ndarray:
    BASELINE_ERROR: float = 0.8 # The maximum value for the error
    
    def err_calc(best_direction: Direction, x: np.ndarray = x, cvar: AttrDict = cvar):
        # Error is the baseline value scaled by the inverse of the reward in the best direction
        err: np.ndarray = x # Set the error to the post to reset all values other than the best direction
        # New error scaling equation
        # f(x) = {
        #   E / x + 0.0.1; x >= 50
        #   sqrt(-(x - 50)) / 8 + 0.25; 1 <= x < 50
        #}
        if cvar.reward >= 50:
            ERROR_CURVE = 12.0
            err_scale: float = ERROR_CURVE / cvar.reward + 0.01
        else:
            err_scale: float = math.sqrt(-(cvar.reward - 50)) / 8 + 0.25
        # err_scale: float = (ERROR_CURVE / cvar.reward) # Factor to scale error by
        err_val: float = (x[best_direction] - BASELINE_ERROR) * err_scale  # Compute the error value for the best direction
        # Set the direction we want to go in to the computed error value
        err[best_direction] = err_val
        # Prevent the error from going beyond the baseline error value
        err = err.clip(-BASELINE_ERROR, BASELINE_ERROR)
        return err

    # Get the best path to the goal
    path: list[Point] = cvar.arena.distance()
    # Compute the best direction based on the best path
    delta_dist: Point = Point(0,0) if len(path) < 2 else path[1] - cvar.arena.player.point
    best_direction: Direction = Point(-np.sign(delta_dist.x), 0).asdirection() if abs(delta_dist.x) == 22 else delta_dist.asdirection()

    # Compute the error for an early return
    err = err_calc(best_direction)

    # Only update the error when the agent has performed an action
    if not cvar.action_performed:
        return err
    cvar.action_performed = False
    log.debug(f'Best Direction is: {best_direction.name}')
    log.debug(f'Best Path: {path[1:-1]}')

    # Compute the reward value for the current timestep
    def reward(player: Player, player_moved: bool, path: list[Point]) -> float:
        # Reward and punishment values
        MOVEMENT_REWARD: float = 1.0
        HALT_PENALTY: float = 5.0
        TOWARDS_GOAL_REWARD: float = 2.0
        AWAY_FROM_GOAL_PENALTY: float = 2.0
        VERY_AWAY_FROM_GOAL_PENALTY: float = 3.0
        HIT_WALL_PENALTY: float = 5.0
        REPEAT_POSITIONS_PENALTY: float = 2.0
        MANY_REPEAT_POSITIONS_PENALTY: float = 4.0

        # The number of steps to the goal (minus the current [0] and goal [len - 1] locations)
        dist_to_goal: int = len(path) - 2
        # How many steps did the agent needd to get to the goal in the last step
        previous_distance: int = len(cvar.arena.distance(start=player.positions[-1])) - 2
        # Did the agent move towards the goal in this timestep
        moved_towards_goal: bool = dist_to_goal < previous_distance
        # Did the agent hit a wall in this timestep
        hit_wall: bool = player.positions[-1] == player.point

        # If the player did not move no other reward calculation is needed
        if not player_moved:
            return -HALT_PENALTY

        # Reward the agent for moving and punish for not moving
        reward: float = MOVEMENT_REWARD if player else 0.0
        
        # Reward the agent for moving towards the goal
        if moved_towards_goal:
            reward += TOWARDS_GOAL_REWARD
            # Reset the counter for moving away from the goal
            cvar.moved_away_from_goal = 0
        else:
            # Increment the counter
            cvar.moved_away_from_goal += 1
            # If the last 2 movements moved away from the goal
            if cvar.moved_away_from_goal == 2:
                # Punish it more
                reward -= VERY_AWAY_FROM_GOAL_PENALTY
                cvar.moved_away_from_goal = 0
            else:
                reward -= AWAY_FROM_GOAL_PENALTY
        # Punish the agent for moving/hitting a wall
        if hit_wall:
            reward -= HIT_WALL_PENALTY
        # Punish the agent for returning to the same points
        if len(player.positions) > 5:
            unique_positions: int = len(set(player.positions[-5:]))
            log.debug(f"Unique Positions: {unique_positions}")
            if unique_positions <= 2:
                reward -= REPEAT_POSITIONS_PENALTY
            elif unique_positions <= 4:
                reward -= MANY_REPEAT_POSITIONS_PENALTY
        log.debug(f"Instantaneous Reward: {reward}")
        return reward

    # Adjust the current state based on the current reward
    cvar.reward += reward(cvar.arena.player, cvar.player_moved, path)
    # Ensure that the minimum reward state possible is 1 for the later division to always succeed
    cvar.reward = max(cvar.reward, 1.0)
    # Update the player info about the change in reward
    cvar.arena.player.info.reward = cvar.reward

    # Recompute the error with the updated reward
    err = err_calc(best_direction)
    log.debug(f'Updated error to: {err}')
    log.debug(f'Updated reward to: {cvar.reward}')

    return err

# Get the distance to a wall in every direction starting from the agent
def detection(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    # Get the detection information from the arena
    tmp = cvar.arena.detection().astype(cvar.dtype)
    # Convert the detection distance to a binary value base on if there is or is not a wall in a direction
    return (tmp / 46.0)

def inhibit(t: float, cvar: AttrDict = cvar) -> np.ndarray:
    if cvar.learning:
        return np.zeros((cvar.ensemble_neurons,), dtype=cvar.dtype)
    return -1000*np.ones((cvar.ensemble_neurons,), dtype=cvar.dtype)

def create_model_fpga():
    global model
    # Global model definition for use with NengoGUI
    model = nengo.Network(label='pacman')
    with model:
        bg = nengo.networks.BasalGanglia(dimensions=cvar.output_dimensions)
        thal = nengo.networks.Thalamus(dimensions=cvar.output_dimensions)

        # Ensembles
        pre = nengo.Ensemble(
            n_neurons = cvar.ensemble_neurons,
            dimensions = cvar.input_dimensions,
            neuron_type = cvar.neuron_type,
            label = 'Pre'
        )
        post = nengo.Ensemble(
            n_neurons = cvar.ensemble_neurons,
            dimensions = cvar.output_dimensions,
            neuron_type = cvar.neuron_type,
            label = 'Post'
        )
        err = nengo.Ensemble(
            n_neurons=cvar.ensemble_neurons,
            dimensions=cvar.output_dimensions,
            neuron_type=cvar.neuron_type,
            label='Error'
        )
        
        # Nodes (interaction with simulation)
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
            size_out=cvar.input_dimensions,
            label='Goal Best Direction'
        )
        g_pnt = nengo.Node(
            output=goal_point_distance,
            size_out=2,
            label='Goal Point Distance'
        )
        p_loc = nengo.Node(
            output=player_location,
            size_out=2,
            label='Player Location'
        )
        g_loc = nengo.Node(
            output=goal_location,
            size_out=2,
            label='Goal Location'
        )
        learn_inhibit = nengo.Node(
            output=inhibit,
            size_out=cvar.ensemble_neurons,
            label='Learning Inhibit Node'
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
        # Node to be able to view reward in nengo gui
        nreward = nengo.Node(
            output=lambda x: cvar.reward,
            size_out=1,
            label='Reward'
        )

        # Ensembles
        err = nengo.Ensemble(
            n_neurons=cvar.ensemble_neurons,
            dimensions=cvar.output_dimensions,
            neuron_type=cvar.neuron_type,
            label='Error',
        )

        # Input Connections
        conn_dist_in = nengo.Connection(
            pre=dist_in,
            post=pre,
            label='Distance Input Connection',
        )
        conn_best_in = nengo.Connection(
            pre=best_dir,
            post=pre,
            label='Best Direction Connection'
        )
        # conn_pre_pl = nengo.Connection(
        #     pre = p_loc,
        #     post = pre[:2],
        #     transform=np.ones(2, dtype=cvar.dtype) / 23.0,
        #     label = 'Player Location Input Connection'
        # )
        # conn_pre_gl = nengo.Connection(
        #    pre = g_loc,
        #    post = pre[2:],
        #    transform=np.ones(2, dtype=cvar.dtype) / 23.0,
        #    label = "Goal Location Input Connction"
        # )
        conn_pac_out_bg = nengo.Connection(
            pre = post,
            post = bg.input,
            label = 'Post -> BG Connection'
        )
        conn_pre_post = nengo.Connection(
            pre = pre,
            post = post,
            label = 'Pre -> Post Connection'
        )
        conn_pre_post.learning_rule_type = cvar.learning_rule_type
        conn_err_learn = nengo.Connection(
            pre=err, 
            post=conn_pre_post.learning_rule,
            label='Error -> Learning Rule Connection'
        )

        # Output Filtering Connections
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

        # Inhibit Connections
        conn_inhibit = nengo.Connection(
            pre=learn_inhibit,
            post=err.neurons,
            label='Error Inhibit Connection'
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
            gui.update() # Update the arena representation inside the GUI
            dpg.render_dearpygui_frame() # Render the updated frame to the GUI
            expected_end_time: float = start_time + target_frame_time # Compute how long to wait for the next frame
            time.sleep(max(expected_end_time - time.time(), 0)) # Wait until its time to render the next frame
        log.debug(f'Simulator ran: {sim.n_steps} steps')
    dpg.destroy_context()

def web_gui():
    gui.create_gui()
    
    gui.display_gui() # Display GUI
    while dpg.is_dearpygui_running():
        jobs = dpg.get_callback_queue()
        dpg.run_callbacks(jobs)

        gui.update_text() # Update text boxes in the gui
        if not gvar.in_gui:
            break

        dpg.render_dearpygui_frame() # Render the updated frame to the GUI
        time.sleep(0.1)
    dpg.destroy_context()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if cvar.log_level == int(logging.NOTSET):
        import yappi
        yappi.set_clock_type('cpu')
        yappi.start()
    # Allow changing the logging level by command line parameter
    if len(sys.argv) > 1:
        if '--nengo' in sys.argv:
            if not gvar.disable_gui:
                # Start the gui in another thread
                gvar.in_gui = True
                t = threading.Thread(target=web_gui, name='GUI')
                t.start()
            # Start the nengo web gui in the main thread
            g = nengo_gui.InteractiveGUI(
                nengo_gui.guibackend.ModelContext(filename=__file__), 
                nengo_gui.guibackend.GuiServerSettings(('0.0.0.0', 8080)),
                nengo_gui.page.PageSettings()
            )
            webbrowser.open(str(g.server.get_url(token="one-time")))
            g.start()

            if not gvar.disable_gui:
                gvar.in_gui = False
                t.join(5)

            if cvar.log_level == int(logging.NOTSET):
                yappi.stop()
                threads = yappi.get_thread_stats()
                for thread in threads:
                    log.info(f'Logging stats for thread: {thread.name}')
                    yappi.get_func_stats(ctx_id=thread.id).print_all()

            # Ensure all references to shared memory are removed before exiting
            del cvar
            del gvar
            del shared.SharedArena.shared_path_cache
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
        if gvar.disable_gui:
            raise ImportError('Failed to load local GUI library')
        main()
    except Exception as e:
        log.critical(f"ERROR: {e}", exc_info=e)
        exit(1)

if '__page__' in locals():
    logging.root.handlers = []
    smart_agent.handle.setFormatter(logging.Formatter(f'{threading.current_thread().name}[{{name}}] | {{levelname}} -> {{message}}', style='{'))
    # Check if the local gui is running and set flag
    for t in threading.enumerate():
        if t.name == 'GUI':
            log.info('Found gui')
            gvar.in_gui = True
            break
    nengo.logger.setLevel(logging.WARNING)
    log.info('Setting up for NengoGUI')
    # Create the model
    global model
    model: nengo.Network = None
    create_model_fpga()

    # Hook that is exectuted every time the simulation is started from the NengoGUI
    def on_start(sim: nengo.Simulator):
        gvar.seed = sim.seed
        gvar.start_time = time.time()
        gvar.run_timer = True

    # Hook that is executed every time the simulation is paused
    def on_pause(sim: nengo.Simulator):
        gvar.end_time = time.time()
        gvar.offset_start_time = time.time()
        gvar.run_timer = False

    # Hook that executes when the simulation is resumed
    def on_continue(sim: nengo.Simulator):
        gvar.offset_time += time.time() - gvar.offset_start_time
        gvar.run_timer = True

    # Hook that is run on every step the model computes (dt = 0.01s ~= 100 steps per sim second)
    def on_step(sim: nengo.Simulator):
        if sim is not None:
            gvar.sim_time = sim.time
            if sim.time == LEARN_STOP_TIME: # Stop learning at 60s in simulation time
                log.info(f"Agent Performance after {round(LEARN_STOP_TIME / 60.0, 2)} mins")
                log.info(cvar.arena.performance)
                cvar.learning = False
            elif sim.time == LEARN_MEASURE_TIME:
                log.info(f"Agent Performance after {round(LEARN_MEASURE_TIME / 60.0, 2)} mins")
                log.info(cvar.arena.performance)            
    
    def on_close(sim: nengo.Simulator):
        log.info(f'Finished simulation after running {sim.n_steps} steps')
        with open(cvar.metrics_file, 'w') as f:
            json.dump(cvar.arena.performance, f, indent=4, cls=vars.JsonEncoder)
        log.info(f'Metrics dumped to {cvar.metrics_file}')