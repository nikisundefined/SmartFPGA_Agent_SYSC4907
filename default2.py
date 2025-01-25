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
error_dimensions = 4

distances: np.ndarray = np.random.randint(low=0, high=23, size=(4))
goal: np.ndarray = np.random.randint(low=0, high=23, size=(2))
current: np.ndarray = np.random.randint(low=0, high=23, size=(2))

flipped: int = 1
prev_state: float = 0.0
rl_state: float = 1.0

MOVEMENT_WEIGHT: float = 0.8
DIRECTION_WEIGHT: float = 0.2

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
    if len(arena.player.positions) == 0:
        return player_location(t)
    return np.array([arena.player.positions[-1].x, arena.player.positions[-1].y], dtype=np.float64)

def delta_distance(t: float) -> np.ndarray:
    tmp = player_location(t) - last_location(t)
    tmp = np.array([-tmp[1], tmp[0], tmp[1], -tmp[0]], dtype=np.float64)
    return np.maximum(tmp, np.zeros(4, dtype=np.float64))

def goal_distance(t: float) -> np.ndarray:
    tmp = player_location(t) - goal_location(t)
    tmp = np.array([-tmp[1], tmp[0], tmp[1], -tmp[0]], dtype=np.float64)
    return np.maximum(tmp, np.zeros(4, dtype=np.float64))

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

    global arena, player_moved, action_performed, rl_state
    index = int(np.argmax(x))
    if math.isclose(x[index],0):
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
        

# Compute the error 4D from 12D input
#    1: Delta Location
#    2: Goal Distance
#    3: Detection Distance
# 
def error(t: float, x: np.ndarray):
    global MOVEMENT_WEIGHT, DIRECTION_WEIGHT
    WEIGHT_ADJUSTMENT: float = 0.01

    ll: np.ndarray = x[0:4]
    gl: np.ndarray = x[4:8]
    sd: np.ndarray = x[8:12]

    if math.isclose(t,int(t)):
        return gl + sd

    if last_action and np.argmax(sd) != int(last_action):
        DIRECTION_WEIGHT += WEIGHT_ADJUSTMENT
        MOVEMENT_WEIGHT -= WEIGHT_ADJUSTMENT
    else:    
        sd *= DIRECTION_WEIGHT
    
    if not ll.any(0):
        MOVEMENT_WEIGHT += WEIGHT_ADJUSTMENT
        DIRECTION_WEIGHT -= WEIGHT_ADJUSTMENT
        gl *= MOVEMENT_WEIGHT

    err = gl + sd
    return err

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

def shutoff(t: float, x) -> np.ndarray:
    for val in x:
        if val > 10:
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
        output=delta_distance,
        size_out=error_dimensions,
        label='Movement Change Input',
    )
    gol_in = nengo.Node(
        output=goal_distance,
        size_out=error_dimensions,
        label='Goal Distance Input'
    )
    mov_out = nengo.Node(
        output=move,
        size_in=output_dimensions,
        label='Movement Output'
    )
    err_tra = nengo.Node(
        output=error,
        size_in=error_dimensions*3,
        size_out=output_dimensions,
        label='Error Compute',
    )
    shutoff_gate = nengo.Node(
        output=limiter,
        label='Shutoff Gate'
    )
    flush = nengo.Node(
        output=shutoff,
        size_in=output_dimensions,
        size_out=output_dimensions,
        label='Post Flush'
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
        post=err_tra[0:4],
        label='Location Delta -> Error Connection'
    )
    conn_goal_loc = nengo.Connection(
        pre=gol_in,
        post=err_tra[4:8],
        transform=-1,
        label='Goal Distance -> Error Connection'
    )
    conn_dist_err = nengo.Connection(
        pre=dist_in,
        post=err_tra[8:12],
        label='Detection Distance -> Error Connection'
    )
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
    conn_flush_post = nengo.Connection(
        pre=flush,
        post=post.neurons,
        transform=-1000000000*np.ones((post.n_neurons, 4)),
        label='Flush -> Post',
    )
    conn_post_flush = nengo.Connection(
        pre=post,
        post=flush,
        label='Post -> Flush',
    )

if __name__ == '__main__':
    import gui
    import time
    import dearpygui.dearpygui as dpg
    import pause

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
            pause.until(expected_end_time)
    dpg.destroy_context()