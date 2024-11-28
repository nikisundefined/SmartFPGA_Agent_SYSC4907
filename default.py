import nengo
import numpy as np
import simulation

arena = simulation.Arena()
last_action: simulation.Direction = None

def pacman_output(time: float):
    VAL_MAX = 20
    detect = arena.detection()
    output = (detect[0] / VAL_MAX, detect[1] / VAL_MAX, detect[2] / VAL_MAX, detect[3] / VAL_MAX)
    return output

def results(time: float, data: np.ndarray):
    if time % 0.1 == 0:
        max_index = np.argmax(np.abs(data))
        #if abs(data[max_index]) > 0.8:
        dir = simulation.Direction(max_index)
        last_action = dir
        print(f"Direction: {dir.name}")
        arena.move(dir)
        print(f"Player position: ({arena.player.x}, {arena.player.y})")
        print(f"Goal position: ({arena.goal.x}, {arena.goal.y})")

def err(data: np.ndarray):
    y_up_delta = (1 if arena.player.y - arena.goal.y < 0 else 0) - (1 if last_action == simulation.Direction.UP and arena.grid[arena.player.y - 1][arena.player.x] == arena.WALL else 0)
    y_down_delta = (1 if arena.player.y - arena.goal.y > 0 else 0) - (1 if last_action == simulation.Direction.DOWN and arena.grid[arena.player.y + 1][arena.player.x] == arena.WALL else 0)
    x_left_delta = (1 if arena.player.x - arena.goal.x > 0 else 0) - (1 if last_action == simulation.Direction.LEFT and arena.grid[arena.player.y][arena.player.x - 1] == arena.WALL else 0)
    x_right_delta = (1 if arena.player.x - arena.goal.x < 0 else 0) - (1 if last_action == simulation.Direction.RIGHT and arena.grid[arena.player.y][arena.player.x + 1] == arena.WALL else 0)
    return np.array([y_up_delta, x_right_delta, y_down_delta, x_left_delta], np.int8)

model = nengo.Network(label='pacman', seed=0)
with model:
    stim = nengo.Node(
        output=pacman_output, 
        size_out=4, 
        label='Pacman Input'
    )
    out = nengo.Node(
        output=results, 
        size_in=4, 
        label='Pacman Output'
    )
    
    a = nengo.Ensemble(
        n_neurons=100, 
        dimensions=4,
        label='Input Layer'
    )
    b = nengo.Ensemble(
        n_neurons=100, 
        dimensions=4,
        label='Output Layer'
    )
    
    error = nengo.Ensemble(
        n_neurons=100,
        dimensions=4,
        label='Input->Output Error'
    )
    
    in_conn = nengo.Connection(stim, a)
    conn = nengo.Connection(a, b, synapse=0.01)
    out_conn = nengo.Connection(b, out)
    
    nengo.Connection(a, error, function=err)
    nengo.Connection(b, error)
    conn.learning_rule_type = nengo.PES()
    nengo.Connection(error, conn.learning_rule)