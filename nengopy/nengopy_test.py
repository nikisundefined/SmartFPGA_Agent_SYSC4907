import nengo
import nengo.neurons
import numpy as np
import neurons as nengopy

class RectifiedLinearImpl(nengo.neurons.AdaptiveLIF):
    def gain_bias(self, max_rates, intercepts):
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
        print(f'gain_bias(max_rates={max_rates.shape}, intercepts={intercepts.shape})')
        return super().gain_bias(max_rates, intercepts)

    def max_rates_intercepts(self, gain, bias):
        print(f'max_rates_intercepts(max_rates={gain.shape}, intercepts={bias.shape})')
        return super().max_rates_intercepts(gain, bias)
    
    def step(self, dt, J, output, **state):
        print(f"step(dt={dt}, J={J.shape}, output={output.shape})")
        return super().step(dt, J, output, **state)

if __name__ == '__main__':
    model: nengo.Network = nengo.Network('Base')
    neuron: nengo.neurons.NeuronType = nengopy.SpikingRectifiedLinear()
    with model:
        inp = nengo.Node(output=np.ones(4), size_out=4)
        cent = nengo.Ensemble(100, 4, neuron_type=neuron)
        out = nengo.Node(output=lambda x, y: print(f"{x}:{y}"), size_in=4)
        nengo.Connection(inp, cent)
        nengo.Connection(cent, out)
    with nengo.Simulator(model) as sim:
        sim.run_steps(10)