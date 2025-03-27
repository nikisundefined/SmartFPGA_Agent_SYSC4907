try:
    from . import fpga
    RectifiedLinear = fpga.RectifiedLinear
except:
    import nengo.neurons
    RectifiedLinear = nengo.RectifiedLinear