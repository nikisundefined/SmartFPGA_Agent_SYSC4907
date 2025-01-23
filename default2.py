#!/usr/bin/env python3

import math
import nengo
import numpy as np
from simulation import Arena, Direction, Point

last_action: Direction = None
player_moved: bool = True
ensemble_neurons = 100
learning_rate: float = 5e-6
neuron_type: nengo.neurons.NeuronType = nengo.SpikingRectifiedLinear()
solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=False)
learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate)
input_dimensions = 4
output_dimensions = 4
error_dimensions = 2

distances: np.ndarray = np.random.randint(low=0, high=23, size=(4))
goal: np.ndarray = np.random.randint(low=0, high=23, size=(2))
current: np.ndarray = np.random.randint(low=0, high=23, size=(2))

flipped: int = 1
prev_state: float = 0.0
rl_state: float = 1.0

MOVEMENT_WEIGHT: float = 0.6
GOAL_WEIGHT: float = 0.3
DIRECTION_WEIGHT: float = 0.1

arena: Arena = Arena()
gui_callback = None
action_performed: bool = False

# returns the current location of the player
def player_location(t: float) -> np.ndarray:
    global arena
    return np.array([arena.player.x, arena.player.y], dtype=np.float64)

def goal_location(t: float) -> np.ndarray:
    global arena
    return np.array([arena.goal.x, arena.goal.y], dtype=np.float64)

def last_location(t: float) -> np.ndarray:
    global arena
    return np.array([arena.player.positions[-1].x, arena.player.positions[-1].y], dtype=np.float64)

# moves the player based on the movement vector (2D)
# Process:
#   Determine how to move (x = + or -, or y = + or -)
#   Move the player
#   If the player has reached the goal
#       Move the goal
def move(t: float, x: np.ndarray):
    if not math.isclose(t, int(t)):
        return
    print(f"Move at {round(t, 2)}")

    global arena, player_moved, action_performed
    index = int(np.argmax(x))
    if math.isclose(x[index],0,rel_tol=1e-1):
        print(f"  No action selected")
        player_moved = False
        return
    
    assert index >= 0 and index <= 3, f"ERROR: Index out of range: {index}"
    tmp = Point(arena.player.x, arena.player.y)
    arena.move(index)

    if player_moved and tmp == arena.player:
        print(f"  Player has stopped moving at {arena.player}")
    player_moved = tmp != arena.player

    print(f"  Direction {Direction(index).name} {x}")
    print(f"  Player Location: {arena.player} | Goal Location: {arena.goal}")

    if arena.on_goal():
        print("  Agent reached the goal")
        arena.set_goal()
    if gui_callback:
        gui_callback()
    action_performed = True
        

# Expand the error signal dimensions:
#   Place the current state and previous state in the last 2 dimentions
def error(t: float, x: np.ndarray):
    RL_MOVE_REWARD: float = -0.01
    RL_MOVE_PUNISH: float = 0.001
    RL_STOPPED_PUNISH: float = 0.002
    RL_REPEATED_PUNISH: float = 0.01
    global arena, last_action, player_moved, prev_state, rl_state, flipped
    # Early return to only update error every ~1 second
    if not math.isclose(t, int(t)):
        return flipped * np.array([x[1], -x[0], -x[1], x[0]], dtype=np.float64) * rl_state

    prev_state = rl_state # save the previous learning value

    # Punish moving into walls/not moving
    if not player_moved:
        rl_state += RL_STOPPED_PUNISH

    # If the the player has travelled in a loop or is stuck on a wall
    if len(arena.player.positions) > 8 and len(set(arena.player.positions[-8:])) == 1:
        print("  Flipping")
        arena.player.positions.clear() # Clear the positions to reset the count
        #flipped *= -1 # invert the value of flipped
        rl_state += RL_REPEATED_PUNISH # Punish the model for this action

    # Compute reward/punishment for going in the right/wrong direction
    if last_action == Direction.UP:
        if arena.player.y > arena.goal.y:
            rl_state += RL_MOVE_REWARD # Reward going towards the goal
        else:
            rl_state += RL_MOVE_PUNISH # Punish going away from the goal
    elif last_action == Direction.DOWN:
        if arena.player.y < arena.goal.y:
            rl_state += RL_MOVE_REWARD
        else:
            rl_state += RL_MOVE_PUNISH
    elif last_action == Direction.LEFT:
        if arena.player.x > arena.goal.x:
            rl_state += RL_MOVE_REWARD
        else:
            rl_state += RL_MOVE_PUNISH
    elif last_action == Direction.LEFT:
        if arena.player.x < arena.goal.x:
            rl_state += RL_MOVE_REWARD
        else:
            rl_state += RL_MOVE_PUNISH
    
    rl_state = max(0.01, rl_state)

    return flipped * np.array([x[1], -x[0], -x[1], x[0]], dtype=np.float64) * rl_state
    # check if the agent is moving towards the goal based on {last_action}
    # check that the {last_action} did not try to move the agent into a wall
    # TODO future: check if the action have been performed repeatedly/looping

def step(t: float) -> np.ndarray:
    global arena
    # if math.isclose(t, int(t)):
    return arena.detection().astype(np.float64)
    # return np.zeros((4,), dtype=np.float64)

def limiter(t: float):
    global action_performed, flipped, rl_state
    if action_performed:
        action_performed = False
        return 1
    return 0

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

# Goal: choose the direction with the largest value in the direction of the goal
# nparray with 4 random values (detection distance)
# nparray with goal location
# nparray with current location
# error is computed as the differece in current and goal positions
# want to choose the direction that has the largest value in the direction which minimizes error

model = nengo.Network(label='pacman')
with model:
    bg = nengo.networks.BasalGanglia(dimensions=output_dimensions)
    thal = nengo.networks.Thalamus(dimensions=output_dimensions)

    # Nodes (interaction with simulation)
    dist_in = nengo.Node(
        output=step,
        size_out=input_dimensions,
        label='Distance Input Node'
    )
    cur_in = nengo.Node(
        output=player_location,
        size_out=error_dimensions,
        label='Current Player Location Input',
    )
    gol_in = nengo.Node(
        output=goal_location,
        size_out=error_dimensions,
        label='Current Goal Location Input'
    )
    mov_out = nengo.Node(
        output=move,
        size_in=output_dimensions,
        label='Movement Output'
    )
    err_tra = nengo.Node(
        output=error,
        size_in=error_dimensions,
        size_out=output_dimensions,
        label='Error Transform',
    )
    shutoff_gate = nengo.Node(
        output=limiter,
        label='Shutoff Gate'
    )

    # Ensembles
    pre = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=input_dimensions,
        neuron_type=neuron_type,
        label='Pre',
    )
    post = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=output_dimensions,
        neuron_type=neuron_type,
        label='Post',
    )
    err = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=error_dimensions,
        neuron_type=neuron_type,
        label='Error',
    )

    # Processing Connections
    conn_dist_in = nengo.Connection(
        pre=dist_in,
        post=pre,
        function=lambda x: x - 12,
        label='Distance Input Connection',
    )
    conn_pre_post = nengo.Connection(
        pre=pre,
        post=post,
        learning_rule_type=learning_rule_type,
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
    conn_play_loc = nengo.Connection(
        pre=cur_in,
        post=err,
        label='Player Location -> Error Connection'
    )
    conn_goal_loc = nengo.Connection(
        pre=gol_in,
        post=err,
        transform=-1,
        label='Goal Location -> Error Connection'
    )
    conn_err_tra = nengo.Connection(
        pre=err,
        post=err_tra,
        label='Error Transformation Connection'
    )
    conn_learn = nengo.Connection(
        pre=err_tra,
        post=conn_pre_post.learning_rule,
        label='Learning Connection'
    )

    conn_shutoff = nengo.Connection(
        pre=shutoff_gate, 
        post=post.neurons, 
        transform=-1000*np.ones((post.n_neurons, 1)), 
        synapse=None,
        label='Limiter'
    )

if __name__ == '__main__':
    import gui
    import time
    import dearpygui.dearpygui as dpg

    target_frame_rate = 30

    gui.create_gui(arena)
    with nengo.Simulator(model, dt=(1.0/target_frame_rate)) as sim:
        dpg.add_text(f"{sim.seed}", tag='seed', parent='Pacman')
        
        gui.display_gui()
        while dpg.is_dearpygui_running():
            start_time = time.time()
            sim.step()
            gui.update_text()
            dpg.set_item_pos('seed', [dpg.get_viewport_width()/2-dpg.get_item_rect_size('seed')[0]/2, 265])
            gui.update_grid(arena)
            dpg.render_dearpygui_frame()
            expected_end_time = start_time + (1.0 / target_frame_rate)
            if time.time() < expected_end_time:
                time.sleep(expected_end_time - time.time())
    dpg.destroy_context()

    # from gui import create_texture
    # import time

    # target_frame_rate = 30

    # rows = 23
    # columns = 23
    # block_size: int = 10
    # width = columns * block_size
    # height = rows * block_size

    # # Updates the grid text with the current state of the arena
    # def update_grid(tag: str | int = 'Environment'):
    #     global arena
    #     texture_data: list[float] = []
    #     _map: dict[int, list[float]] = {
    #         Arena.EMPTY: [0, 0, 0, 0], # Black
    #         Arena.WALL: [0, 0, 255 / 255, 255 / 255], # Blue
    #         Arena.PLAYER: [255/ 255, 255 / 255, 0, 255/ 255], # Yellow
    #         Arena.GOAL: [0, 255 / 255, 0, 255 / 255], # Green
    #     }

    #     for y in range(arena.m): # Every Y coordinate
    #         for _ in range(block_size): # Every Y block
    #             for x in range(arena.n): # Every X coordinate
    #                 for _ in range(block_size): # Every X block
    #                     # RGBA pixel format
    #                     texture_data.extend(_map[arena.grid[y][x]]) # Pixel value
    #     dpg.set_value(item=tag, value=texture_data)

    # dpg.create_context()
    # dpg.create_viewport(title='Pacman [float]', width=width + 100, height=height + 100)
    # dpg.setup_dearpygui()

    # arena_texture: list[float] = create_texture(rows * block_size, columns * block_size)
    # with dpg.texture_registry(show=False):
    #     dpg.add_dynamic_texture(width=width, height=height, default_value=arena_texture, tag='Environment')
    # update_grid()

    # with dpg.window(tag="Pacman"):
    #     dpg.add_image("Environment", width=width, height=height)

    # dpg.show_viewport()
    # dpg.set_primary_window("Pacman", True)
    # with nengo.Simulator(model, seed=0, dt=(1.0/target_frame_rate)) as sim:
    #     while dpg.is_dearpygui_running():
    #         start_time = time.time()
    #         sim.step()
    #         update_grid()
    #         dpg.render_dearpygui_frame()
    #         expected_end_time = start_time + (1.0 / target_frame_rate)
    #         if time.time() < expected_end_time:
    #             time.sleep(expected_end_time - time.time())
    # dpg.destroy_context()