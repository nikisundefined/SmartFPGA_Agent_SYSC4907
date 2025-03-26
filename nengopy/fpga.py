import struct
import os.path
import multiprocessing
from typing import Optional

import numpy as np
from pynq import DefaultIP
from pynq import Overlay
from pynq import allocate
from pynq import PL
from pynq.buffer import PynqBuffer
from pynq.lib.dma import DMA

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

    def __init__(self, amplitude: float, hls_ip: 'FPGADriver', dma: DMA):
        self.amplitude = amplitude
        self.ip = hls_ip
        self.dma = dma
        self.buffer: PynqBuffer = None

    def _allocate(self, elems) -> None:
        with RectifiedLinearFPGA.lock:
            if self.buffer is not None:
                del self.buffer
            self.buffer = allocate(shape=(elems, 4), dtype=np.float64)
    
    def _load_buffer(self, function: int, arg0: np.ndarray, arg1: Optional[np.ndarray] = None, arg2: Optional[np.ndarray] = None) -> None:
        # Ensure there is enough space in the buffer
        if self.buffer is None or arg0.size < self.buffer.shape[0]:
            self._allocate(arg0.size)
        # Copy elements into the buffer
        self.buffer[:, 0] = function
        self.buffer[:, 1] = arg0.ravel()
        # Only put arg1 and arg2 if they were specified
        if arg1 is not None:
            self.buffer[:, 2] = arg1.ravel()
        if arg2 is not None:
            self.buffer[:, 3] = arg2.ravel()

    def _run(self) -> None:
        # Transfer the buffer address to the DMA channels
        self.dma.recvchannel.transfer(self.buffer)
        self.dma.sendchannel.transfer(self.buffer)
        # Start the accelerator
        self.ip.write(0x0, 0x1)

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

            output[...] = self.buffer[:, 1]
        

class FPGADriver(DefaultIP):

    def __init__(self, description):
        super().__init__(description)

    bindto = ['xilinx.com:hls:nengofpga:1.0']

    def rectified_linear(self, dma: DMA) -> RectifiedLinearFPGA:
        return RectifiedLinearFPGA(1, self, dma)

if __name__ == '__main__':
    PL.reset() # Reset any cached versions of the bitstream and hardware info
    bitstream_path: str = '/home/xilinx/nengofpga/nengofpga.bit'
    print("Programming the FPGA")
    print("Path to bitstream")
    tmp: str = input(f'[{bitstream_path}]: ')
    if len(tmp) != 0: bitstream_path = tmp
    if not bitstream_path.endswith('.bit'):
        raise ValueError('Not a bitstream file')
    if not os.path.exists(bitstream_path):
        raise FileNotFoundError(f'Bitstream file {bitstream_path} does not exist')
    ol = Overlay(bitstream_path)
    print("FPGA programmed")

    if 'nengofpga_0' not in ol.ip_dict.keys() or 'ReadWriteDMA' not in ol.ip_dict.keys():
        raise RuntimeError('NengoFPGA or DMA not found')
    hls_ip: FPGADriver = ol.nengofpga_0
    dma_rw: DMA = ol.ReadWriteDMA
    neuron: RectifiedLinearFPGA = hls_ip.rectified_linear(dma_rw)


# PL.reset() # Reset any cached versions of the bitstream and hardware info
# bitstream_path: str = '/home/xilinx/nengofpga/nengofpga.bit'
# ol = Overlay(bitstream_path)
# ol.ip_dict.keys()
# hls_ip: DefaultIP = ol.nengofpga_0
# hls_ip.register_map
# dma_rw: DMA = ol.ReadWriteDMA
# dma_rw.register_map
# fpga_input: PynqBuffer = allocate(shape=(4,4), dtype=np.float64)
# fpga_output: PynqBuffer = allocate(shape=(4,4), dtype=np.float64)
# fpga_input[:, 1:-1] = np.random.rand(4, 2)
# fpga_input
# dma_rw.recvchannel.transfer(fpga_output)
# dma_rw.sendchannel.transfer(fpga_input)
# fpga_output
