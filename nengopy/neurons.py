try:
    from . import fpga
    RectifiedLinear = fpga.RectifiedLinear
except:
    import nengo
    RectifiedLinear = nengo.RectifiedLinear