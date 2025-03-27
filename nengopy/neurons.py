try:
    from . import fpga
    RectifiedLinear = fpga.RectifiedLinear
except ImportError:
    import nengo.neurons
    RectifiedLinear = nengo.RectifiedLinear