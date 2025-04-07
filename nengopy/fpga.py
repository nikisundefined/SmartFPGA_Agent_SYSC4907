import os.path
import multiprocessing
import time
import logging
import fcntl
import atexit

import numpy as np
from pynq import DefaultIP
from pynq import Overlay
from pynq import allocate
from pynq import PL
from pynq.buffer import PynqBuffer
from pynq.lib.dma import DMA

import nengo
import nengo.neurons

log: logging.Logger = logging.getLogger('nengopy.fpga')
LOCK_FILE: str = 'fpga.lock'

# Notes:
#   DMA channel seems to fail if any transfer fails 
#       (either due to a lack of output space or insufficient inputs) 
#       and cannot be restarted without fully restarting the board
#   If the DMA channel triggers ANY interrupt that is not handled by the system,
#       the channel locks up until the system is reset (Any interrupt specified under dma_rw.register_map)
#   IP MUST!! generate TLAST signal otherwise DMA and Pynq have no way of knowing when the IP is finished, 
#       results is buffers stalling while waiting for something to happen
#   Writes to the IP that are not written in the register_map but are written in the synthesis are valid, 
#       however there is not read access to the field so the value must be tracked internally
#   IP seems to stall when directly writing to the output packet without first copying the return value to another location
#   Arbitrary sizes of input and output packets can be used however,
#       since the size cannot be changed, the appropriately sized packet must also be returned

class RectifiedLinearFPGA:
    RectifiedLinear_GainBias = 0
    RectifiedLinear_MaxRateIntercepts = 1
    RectifiedLinear_Step = 2

    lock = multiprocessing.RLock()
    instance: 'RectifiedLinearFPGA' = None

    def __init__(self, amplitude: float, hls_ip: 'FPGADriver', dma: DMA):
        self.amplitude = amplitude
        self.ip = hls_ip
        self.dma = dma
        self.buffer: PynqBuffer = None
        log.info('Hello from RectifiedLinearFPGA')

    def _allocate(self, elems: int) -> None:
        with RectifiedLinearFPGA.lock:
            if self.buffer is not None:
                del self.buffer
            self.buffer = allocate(shape=(elems, 4), dtype=np.float64)
            log.debug(f'Allocated FPGA buffer with size {self.buffer.size}')
    
    def _load_buffer(self, function: int, arg0, arg1 = None, arg2 = None) -> None:
        def to_array(x):
            """Convert int, float, or np.ndarray to a numpy array."""
            if isinstance(x, (int, float)):
                return x
            elif isinstance(x, np.ndarray):
                return x.ravel()
            else:
                raise TypeError(f"Invalid type {type(x)} for buffer input")

        arg0 = to_array(arg0)
        arg1 = to_array(arg1) if arg1 is not None else None
        arg2 = to_array(arg2) if arg2 is not None else None

        # Allocate buffer if not already allocated or not large enough
        max_size = max(arg0.size if type(arg0) is np.ndarray else 1, arg1.size if arg1 is not None else 0, arg2.size if arg2 is not None else 0)
        if self.buffer is None or self.buffer.shape[0] < max_size:
            self._allocate(max_size)

        self.buffer[:, 0] = function
        self.buffer[:, 1] = arg0
        if arg1 is not None:
            self.buffer[:arg1.size, 2] = arg1
        if arg2 is not None:
            self.buffer[:arg2.size, 3] = arg2

    def _run(self) -> None:
        # Transfer the buffer address to the DMA channels
        self.dma.recvchannel.transfer(self.buffer)
        self.dma.sendchannel.transfer(self.buffer)
        # Start the accelerator
        self.ip.write(0x0, 0x1)

        # Wait for both MM2S_DMASR.Idle and S2MM_DMASR.Idle to be 1
        while not (self.dma.register_map.MM2S_DMASR.Idle and self.dma.register_map.S2MM_DMASR.Idle):
            time.sleep(0.001)  # Sleep briefly to prevent excessive CPU usage

    def gain_bias(self, max_rates: np.ndarray, intercepts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the gain and bias needed to satisfy max_rates, intercepts.

        This takes the neurons, approximates their response function, and then
        uses that approximation to find the gain and bias value that will give
        the requested intercepts and max_rates.

        Determine gain and bias by shifting and scaling the lines.

        Parameters
        ----------
        max_rates : (n_neurons,) array_like
            Maximum firing rates of neurons.
        intercepts : (n_neurons,) array_like
            X-intercepts of neurons.

        Returns
        -------
        gain : (n_neurons,) array_like
            Gain associated with each neuron. Sometimes denoted alpha.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.
        """
        assert max_rates.shape == intercepts.shape, f"RectifiedLinearFPGA::gain_bias -> max_rates and intercepts must have the same shape | {max_rates.shape} != {intercepts.shape}"
        # Obtain views on the input arguments in with the corrent datatype
        max_rates = np.array(max_rates, ndmin=1, copy=False, dtype=np.float64)
        intercepts = np.array(intercepts, ndmin=1, copy=False, dtype=np.float64)

        with RectifiedLinearFPGA.lock: # Aquire the lock on the FPGA just incase
            # Load buffer with the arguments in the required structure
            self._load_buffer(RectifiedLinearFPGA.RectifiedLinear_GainBias, max_rates, intercepts)

            # Compute the result
            self._run()

            # Copy the results out of the buffer
            gain = np.array(self.buffer[:, 1], ndmin=1, copy=True, dtype=np.float64)
            bias = np.array(self.buffer[:, 2], ndmin=1, copy=True, dtype=np.float64)
        return gain, bias
    
    def max_rates_intercepts(self, gain: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the max_rates and intercepts given gain and bias.

        Compute the inverse of gain_bias.

        Parameters
        ----------
        gain : (n_neurons,) array_like
            Gain associated with each neuron. Sometimes denoted alpha.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.

        Returns
        -------
        max_rates : (n_neurons,) array_like
            Maximum firing rates of neurons.
        intercepts : (n_neurons,) array_like
            X-intercepts of neurons.
        """
        assert gain.shape == bias.shape, f"RectifiedLinearFPGA::max_rates_intercepts -> gain and bias must have the same shape | {gain.shape} != {bias.shape}"
        
        gain = np.array(gain, ndmin=1, copy=False, dtype=np.float64)
        bias = np.array(bias, ndmin=1, copy=False, dtype=np.float64)

        with RectifiedLinearFPGA.lock:
            self._load_buffer(RectifiedLinearFPGA.RectifiedLinear_MaxRateIntercepts, gain, bias)

            self._run()

            max_rates = np.array(self.buffer[:, 1], ndmin=1, copy=True, dtype=np.float64)
            intercepts = np.array(self.buffer[:, 2], ndmin=1, copy=True, dtype=np.float64)
        return max_rates, intercepts
    
    def step(self, dt, J, output, *, amplitude: float | None = None) -> None:
        """
        Implements the differential equation for this neuron type.
        
        Implement the rectification nonlinearity.

        Parameters
        ----------
        dt : float
            Simulation timestep.
        J : (n_neurons,) array_like
            Input currents associated with each neuron.
        output : (n_neurons,) array_like
            Output activity associated with each neuron (e.g., spikes or firing rates).
        state : {str: array_like}
            State variables associated with the population.
        """
        J = np.array(J, ndmin=1, copy=False, dtype=np.float64)

        with RectifiedLinearFPGA.lock:
            self._load_buffer(RectifiedLinearFPGA.RectifiedLinear_Step, amplitude, J)

            self._run()

            output[:] = self.buffer[:output.size, 2].reshape(output.shape)
        

class FPGADriver(DefaultIP):

    def __init__(self, description):
        super().__init__(description)

    bindto = ['xilinx.com:hls:nengofpga:1.0']

    def rectified_linear(self, dma: DMA) -> RectifiedLinearFPGA:
        if not RectifiedLinearFPGA.instance:
            RectifiedLinearFPGA.instance = RectifiedLinearFPGA(1, self, dma)
        return RectifiedLinearFPGA.instance
    
class RectifiedLinear(nengo.neurons.RectifiedLinear):
    def __init__(self, **kwargs):
        super().__init__(*kwargs)
    
    def gain_bias(self, max_rates, intercepts):
        return neuron.gain_bias(max_rates, intercepts)
    
    def max_rates_intercepts(self, gain, bias):
        return neuron.max_rates_intercepts(gain, bias)

    def step(self, dt, J, output):
        neuron.step(dt, J, output, amplitude=self.amplitude)

# Check if the overlay has already been loaded
def aquire_lock() -> bool:
    fd = None
    try:
        fd = open(LOCK_FILE, 'w')
        fcntl.flock(fd, fcntl.LOCK_NB | fcntl.LOCK_EX)
        atexit.register(lambda: os.remove(LOCK_FILE))
        return True
    except (OSError, IOError):
        if fd: os.close(fd)
        return False

download: bool = aquire_lock()
bitstream_path: str = '/home/xilinx/nengofpga/nengofpga.bit'
ol = Overlay(bitstream_path, download=download)
hls_ip: FPGADriver = ol.nengofpga_0
dma_rw: DMA = ol.ReadWriteDMA
neuron: RectifiedLinearFPGA = hls_ip.rectified_linear(dma_rw)
neuron_type: RectifiedLinear = RectifiedLinear()

if __name__ == '__main__':
    print("Generating Model")
    with nengo.Network('TestNet') as net:
        pre = nengo.Ensemble(
            n_neurons = 400, 
            dimensions = 4,
            neuron_type = neuron_type,
            label='Pre'
        )
        post = nengo.Ensemble(
            n_neurons=400,
            dimensions=4,
            neuron_type=neuron_type,
            label='Post'
        )
        nengo.Connection(pre, post)

        print("Preparing Simulator")
        with nengo.Simulator(net) as sim:
            print(f"Running sim at: {sim.time}")
            sim.run(1)
            time.sleep(5)
