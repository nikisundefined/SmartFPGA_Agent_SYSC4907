import math
import nengo
import nengo.networks
import numpy as np
import lib.simulation as simulation
from lib.simulation import Arena

# NOTE: Old version of model before refactor

arena = simulation.Arena()
last_action: simulation.Direction = None
ensemble_neurons = 100
learning_rate: float = 5e-6
neuron_type: nengo.neurons.NeuronType = nengo.SpikingRectifiedLinear()
solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=True)
learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate)
input_dimensions = 23*23
output_dimensions = 4

# Inputs: Distance to wall in all 4 directions from player
#         Absolute Distance to goal from player
#         *Absolute Distance to the ghost(s)

# Output: The direction the player should move (Evaluated/Performed every 1s)

# TODO: Implement Random Goal Positions
#       Implement Ghosts

# New TODO:
#       Implement simple 1D simulation and model
#       Adapt NengoFPGA RL code into custom simulation
#       Learn how the error function affects the models behaviour

def pacman_input_distance(t: float) -> np.ndarray:
    tmp: list[float] = arena.detection().astype(np.float32).tolist() # as format [up, down, left, right]
    tmp[0] = (tmp[0] / arena.m)
    tmp[1] = (tmp[1] / arena.m)
    tmp[2] = (tmp[2] / arena.n)
    tmp[3] = (tmp[3] / arena.n)
    return np.asarray(tmp, np.float32)

def pacman_input_goal(t: float) -> np.ndarray:
    # Return spike in best direction(s)
    goal_pos: simulation.Point = arena.goal
    player_pos: simulation.Point = arena.player
    return np.asarray([ # format: [up, down, left, right]
        1 if goal_pos.y > player_pos.y else 0,
        1 if goal_pos.y < player_pos.y else 0,
        1 if goal_pos.x < player_pos.x else 0,
        1 if goal_pos.x > player_pos.x else 0,
    ], np.float32)
    return np.asarray([1 - (arena.distance() / math.sqrt(arena.n ** 2 + arena.m ** 2))], np.float32)

def pacman_output(t: float, x: np.ndarray):
    if t % 0.5 == 0:
        dir: simulation.Direction = simulation.Direction(np.argmax(np.abs(x)))
        arena.move(dir)
        print(f'Output at {t}: {dir.name}')
        print(f'Player pos: ({arena.player.x},{arena.player.y})')
        if arena.on_goal():
            print(f'Agent reached the goal at: ({arena.goal.x},{arena.goal.y})')
            arena.set_goal()
            print(f'New goal at ({arena.goal.x},{arena.goal.y})')

def translation_output(t: float, x: np.ndarray):
    if t % 0.5 == 0:
        print(f'Ouptut at {t}: {x}')

model = nengo.Network(label='pacman', seed=0)
with model:
    # Input Layer: Composed of -- The distance in all directions
    #                          -- Absolute distance to the goal
    distance_in = nengo.Node(
        output=pacman_input_distance,
        size_out=4,
        label='Pacman Distance Input'
    )
    goal_in = nengo.Node(
        output=pacman_input_goal,
        size_out=4,
        label='Goal Distance Input'
    )

    # Layers
    input_layer = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        neuron_type=neuron_type,
        dimensions=4,
        label='Input Layer'
    )
    hidden_layer = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        neuron_type=neuron_type,
        dimensions=4,
        label='Hidden Layer'
    )
    output_layer = nengo.networks.BasalGanglia(
        dimensions=4,
        n_neurons_per_ensemble=ensemble_neurons,
        label='Output Layer'
    )
    trans_layer = nengo.Ensemble(
        n_neurons=4,
        neuron_type=neuron_type,
        dimensions=1,
        label='Translation Layer'
    )

    # Output Layer:
    out = nengo.Node(
        output=pacman_output,
        size_in=4,
        label='Network Output'
    )
    trans = nengo.Node(
        output=translation_output,
        size_in=4,
        label='Translation Ouptut'
    )

    # Connections
    in_dist_conn = nengo.Connection(
        pre=distance_in,
        post=input_layer,
        synapse=None,
        label='Distance -> Input'
    )
    in_goal_conn = nengo.Connection(
        pre=goal_in,
        post=input_layer,
        synapse=None,
        label='Goal -> Input'
    )
    in_hid_conn = nengo.Connection(
        pre=input_layer,
        post=hidden_layer,
        solver=solver_type,
        learning_rule_type=learning_rule_type,
        label='Input -> Hidden'
    )
    hid_out_conn = nengo.Connection(
        pre=hidden_layer,
        post=output_layer.input,
        synapse=None,
        label='Hidden -> Output'
    )
    out_pac_conn = nengo.Connection(
        pre=output_layer.output,
        post=out,
        synapse=None,
        label='Output -> Pacman'
    )
    trans_conn = nengo.Connection(
        pre=hidden_layer,
        post=trans_layer.neurons,
        synapse=None,
        label='Hidden -> Translation'
    )
    trans_out_conn = nengo.Connection(
        pre=trans_layer.neurons,
        post=trans,
        synapse=None,
        label='Translation -> Output'
    )

    # Learning
    err_ens = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        neuron_type=neuron_type,
        dimensions=4,
        label='Error'
    )
    err_out_conn = nengo.Connection(
        pre=output_layer.output,
        post=err_ens,
        synapse=None,
    )
    err_in_conn = nengo.Connection(
        pre=goal_in,
        post=err_ens,
        synapse=None,
        transform=-1
    )

    err_in_hid_conn = nengo.Connection(
        pre=err_ens,
        post=in_hid_conn.learning_rule,
        synapse=None
    )
    err_hid_out_conn = nengo.Connection(
        pre=err_ens,
        post=in_hid_conn.learning_rule,
        synapse=None
    )

if __name__ == '__main__':
    pass