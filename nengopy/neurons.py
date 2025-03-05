import numpy as np

import nengopy.nengocpp as nengocpp
import nengo.neurons

class RectifiedLinear(nengocpp.RectifiedLinearImpl, nengo.neurons.RectifiedLinear):
    def __init__(self, **kwargs):
        nengocpp.RectifiedLinearImpl.__init__(self)
        nengo.neurons.RectifiedLinear.__init__(self, **kwargs)

class SpikingRectifiedLinear(nengocpp.SpikingRectifiedLinearImpl, nengo.neurons.SpikingRectifiedLinear):
    def __init__(self, **kwargs):
        nengocpp.SpikingRectifiedLinearImpl.__init__(self)
        nengo.neurons.SpikingRectifiedLinear.__init__(self, **kwargs)

    def current(self, x, gain, bias):
        ret_external = gain * x + bias
        gain = np.broadcast_to(gain, x.shape)
        bias = np.broadcast_to(bias, x.shape)
        return nengo.neurons.NeuronType.current(self, x, gain, bias)

    def rates(self, x, gain, bias):
        # ret_internal = nengocpp.SpikingRectifiedLinearImpl.rates(self, x, gain, bias)
        ret_external = nengo.neurons.SpikingRectifiedLinear.rates(self, x, gain, bias)
        return ret_external