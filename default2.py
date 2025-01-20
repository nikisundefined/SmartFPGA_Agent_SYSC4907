#!/usr/bin/env python3

import math
import time
import nengo
import numpy as np
import simulation

last_action: simulation.Direction = None
ensemble_neurons = 100
learning_rate: float = 5e-6
neuron_type: nengo.neurons.NeuronType = nengo.SpikingRectifiedLinear()
solver_type: nengo.solvers.Solver = nengo.solvers.LstsqL2(weights=True)
learning_rule_type: nengo.learning_rules.LearningRuleType = nengo.learning_rules.PES(learning_rate=learning_rate)
input_dimensions = 23*23
output_dimensions = 4

distances: np.ndarray = np.random.randint(low=0, high=23, size=(4))
goal: np.ndarray = np.random.randint(low=0, high=23, size=(2))
current: np.ndarray = np.random.randint(low=0, high=23, size=(2))

# returns the current location of the player
def location(t: float) -> np.ndarray:
    global current
    return current

# moves the player based on the movement vector (2D)
# Process:
#   Determine how to move (x = + or -, or y = + or -)
#   Move the player
#   If the player has reached the goal
#       Move the goal
def move(t: float, x: np.ndarray):
    global current, goal
    
    # Compute the direction to move
    x_abs = np.abs(x).astype(np.int8)
    x_sgn = np.sign(x).astype(np.int8)
    argmax = np.argmax(x_abs)
    dir = np.ndarray(shape=(2))
    dir[argmax] = x_sgn[argmax]

    # Move the player
    print(f"Moving {dir}")
    current += dir

    # Check if we reached the goal
    if current.astype(np.int8) == goal.astype(np.int8):
        goal = np.random.randint(low=0, high=23, size=(2))
        print("Reached the goal")
        print(f"Goal is now at: {goal}")

# Compute the error signal defined as:
#   The distance between the current location and the goal location
def error(t: float, x: np.ndarray):
    pass # TODO

def step(t: float) -> np.ndarray:
    global distances
    # if int(t) % 2:
    #     distances = np.random.randint(low=0, high=23, size=(4))
    return distances

def output(t: float, x):
    print(np.max(x))

# Goal: choose the direction with the largest value in the direction of the goal
# nparray with 4 random values (detection distance)
# nparray with goal location
# nparray with current location
# error is computed as the differece in current and goal positions
# want to choose the direction that has the largest value in the direction which minimizes error

model = nengo.Network(label='pacman', seed=0)
with model:
    bg = nengo.networks.BasalGanglia(dimensions=output_dimensions)
    thal = nengo.networks.Thalamus(dimensions=output_dimensions)

    inp = nengo.Node(
        output=step,
        size_out=output_dimensions,
        label='Input Node'
    )
    pre = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=output_dimensions,
        neuron_type=neuron_type,
        label='Pre',
    )
    post = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=output_dimensions,
        neuron_type=neuron_type,
        label='Post',
    )
    post_post = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=output_dimensions,
        neuron_type=neuron_type,
        label='Post Post'
    )
    err = nengo.Ensemble(
        n_neurons=ensemble_neurons,
        dimensions=output_dimensions,
        neuron_type=neuron_type,
        label='Error',
    )

    conn_in = nengo.Connection(
        pre=inp,
        post=pre
    )
    conn_pre_bg = nengo.Connection(
        pre=pre,
        post=bg.input,

    )
    conn_bg_thal = nengo.Connection(
        pre=bg.output,
        post=thal.input
    )
    conn_thal_post = nengo.Connection(
        pre=thal.output,
        post=post,
    )
    conn_post_post = nengo.Connection(
        pre=post,
        post=post_post,
        solver=solver_type,
        learning_rule_type=learning_rule_type
    )
    conn_pre_err = nengo.Connection(
        pre=pre,
        post=err,
        transform=-1
    )
    conn_post_err = nengo.Connection(
        pre=post_post,
        post=err,
    )
    conn_learn = nengo.Connection(
        pre=err,
        post=conn_post_post.learning_rule
    )