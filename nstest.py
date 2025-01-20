from matplotlib import pyplot as plt
import nengo
import nengo.connection
import nengo.network
import nengo.probe
import numpy as np
import simulation
from nengo.processes import WhiteSignal
from nengo.solvers import LstsqL2

y = [4,3,13,13,6,4,15,3,2,5,7,9,20,6,15,4,13,14,6,12] # test value
# (0,0) = top left (23,23) = bottem right
model = nengo.Network(label = "north south test")
with model:
    north_south = nengo.networks.BasalGanglia(dimensions = 2)
    thal = nengo.networks.Thalamus(dimensions=2)


# input 4 values between 0 and 23 chose largest distance from wall
# 4 more, goal / current location; (x,y),(x,y)
# compute to chose a cardinal direction


class ActionIterator:
    def __init__(self, dimensions):
        self.actions = np.ones(dimensions) * 0.1

    def step(self, t):
        if (np.random.randint(0,99) > 50): # if y > 12 north is good
            dominate  = 1
        else:
            dominate = 0
        self.actions[:] = -0.2  
        self.actions[dominate] = 0.8
        return self.actions
    
action_iterator = ActionIterator(dimensions = 2)

with model:
    actions = nengo.Node(action_iterator.step, label = "actions")

    nengo.Connection(actions, north_south.input, synapse = None)
    selected_action = nengo.Probe(north_south.output, synapse = 0.01)
    input_actions = nengo.Probe(actions, synapse = 0.01)

    nengo.Connection(north_south.output, thal.input)
    #more testing
    pre = nengo.Ensemble(60, dimensions=2)
    nengo.Connection(pre, north_south.input)
    post = nengo.Ensemble(60, dimensions=2)
    conn = nengo.Connection(post, thal.input)

    #adding in learning
    error = nengo.Ensemble(60, dimensions=2)
    error_p = nengo.Probe(error, synapse=0.03)

    # Error = actual - target = post - pre
    nengo.Connection(thal.output, error)
    nengo.Connection(north_south.input, error, transform=-1)

    # Add the learning rule to the connection
    conn.learning_rule_type = nengo.PES()

    # Connect the error into the learning rule
    nengo.Connection(error, conn.learning_rule)

def inhibit(t):
    return 2.0 if t > 13.0 else 0.0

with model:
    inhib = nengo.Node(inhibit)
    nengo.Connection(inhib, error.neurons, transform=[[-1]] * error.n_neurons)