from matplotlib import pyplot as plt
import nengo
import nengo.connection
import nengo.network
import nengo.probe
import numpy as np
import simulation
from nengo.processes import WhiteSignal
from nengo.solvers import LstsqL2

north = [4,3,5,7,6,4,245,3,2,5,7,9,7,6,5,4,3,54,6,2] # test values
south = [2,5,4,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9] # test values
model = nengo.Network(label = "north south test")
with model:
    north_south = nengo.networks.BasalGanglia(dimensions = 2)


class ActionIterator:
    def __init__(self, dimensions):
        self.actions = np.ones(dimensions) * 0.1

    def step(self, t):
        n_s = north[int(t)] - south[int(t)]
        if (n_s >= 0):
            dominate  = 1
        else:
            dominate = 0
        self.actions[:] = 0.1
        self.actions[dominate] = 0.8
        return self.actions
    
action_iterator = ActionIterator(dimensions = 2)

with model:
    actions = nengo.Node(action_iterator.step, label = "actions")

    nengo.Connection(actions, north_south.input, synapse = None)
    selected_action = nengo.Probe(north_south.output, synapse = 0.01)
    input_actions = nengo.Probe(actions, synapse = 0.01)

    #more testing
    pre = nengo.Ensemble(60, dimensions=2)
    nengo.Connection(north_south.input, pre)
    post = nengo.Ensemble(60, dimensions=2)
    conn = nengo.Connection(post, north_south.output, function=lambda x: np.random.random(2))
