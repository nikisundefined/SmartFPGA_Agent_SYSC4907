import struct
import os.path

import numpy as np
from pynq import DefaultIP
from pynq import Overlay
from pynq import allocate
from pynq.buffer import PynqBuffer
from pynq.lib.dma import DMA

class RectifiedLinearFPGA:
    RectifiedLinear_GainBias = 0
    RectifiedLinear_MaxRateIntercepts = 1
    RectifiedLinear_Step = 2

    def __init__(self, amplitude: float, hls_ip: 'FPGADriver', ol: Overlay):
        self.amplitude = amplitude
        self.ip = hls_ip
        self.dma_rw: DMA = ol.ReadWriteDMA
        self.dma_r0: DMA = ol.ReadDMA0
        self.dma_r1: DMA = ol.ReadDMA1

    def step(self, dt: float, J: np.ndarray, output: np.ndarray) -> None:
        #        output[...] = self.amplitude * np.maximum(0.0, J) # Original Implementation
        fpga_output = allocate(shape=(output.shape[0], 4), dtype=np.float64)
        fpga_J = allocate(shape=J.shape, dtype=np.float64)

        # Load inputs to computation
        fpga_J[:] = J

        # Setup Internal IP registers
        self.ip.write(FPGADriver.Function_Select, RectifiedLinearFPGA.RectifiedLinear_Step)
        self.ip.write(FPGADriver.Amplitude_Low, struct.pack('d', self.amplitude))

        # Send inputs to FPGA
        self.dma_rw.recvchannel.transfer(fpga_output)
        self.dma_r0.sendchannel.transfer(fpga_J)

        # Start the IP
        self.ip.write(0x0, 0x1)

        # Copy output from FPGA buffer to return buffer
        output[:] = fpga_output[..., 0]

    def max_rates_intercepts(self, gain: np.ndarray, bias: np.ndarray) -> tuple:
        fpga_gain : PynqBuffer = allocate(shape=gain.shape, dtype=np.float64)
        fpga_bias: PynqBuffer = allocate(shape=gain.shape, dtype=np.float64)
        fpga_output: PynqBuffer = allocate(shape=(gain.shape[0], 4), dtype=np.float64)

        fpga_gain[:] = gain
        fpga_bias[:] = bias

        self.ip.write(0x0, 0x01)
        self.ip.write(FPGADriver.Function_Select, RectifiedLinearFPGA.RectifiedLinear_GainBias)

        self.dma_rw.recvchannel.transfer(fpga_output)
        self.dma_r1.sendchannel.transfer(fpga_bias)
        self.dma_r0.sendchannel.transfer(fpga_gain)

        return fpga_output[:, 0], fpga_output[:, 1]

class FPGADriver(DefaultIP):
    Function_Select = 0x10
    Amplitude_Low = 0x18
    Amplitude_High = 0x1c

    def __init__(self, description):
        super().__init__(description)

    bindto = ['xilinx.com:hls:nengofpga:1.0']

    def rectified_linear(self, ol: Overlay) -> RectifiedLinearFPGA:
        return RectifiedLinearFPGA(1, self, ol)

if __name__ == '__main__':
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

    if 'nengofpga_0' not in ol.ip_dict.keys():
        raise RuntimeError('NengoFPGA not found')
    hls_ip: FPGADriver = ol.nengofpga_0
    neuron: RectifiedLinearFPGA = hls_ip.rectified_linear(ol)

# Old Script Testing Code
'''
print("Programming the FPGA")
ol = Overlay('/home/xilinx/nengofpga/nengofpga.bit')

print("Inspect all the IP names")
print(ol.ip_dict.keys())

print("Inspect the HLS IP registers")
hls_ip = ol.step_0
print(hls_ip)

dma = ol.axi_dma_0
dma_send = ol.axi_dma_0.sendchannel
dma_recv = ol.axi_dma_0.recvchannel

print("Starting HLS IP")
print(hls_ip.register_map)
CONTROL_REGISTER = 0x0
hls_ip.write(CONTROL_REGISTER, 0x81)


data_size = 5
input_buffer = allocate(shape=(data_size,), dtype=np.float64)
output_buffer = allocate(shape=(data_size,), dtype=np.float64)

a = [i for i in range(data_size)]
a = np.float64(a)
#for i in range(data_size):
#    input_buffer[i] = np.float32(i)
np.copyto(input_buffer, a)

print("Starting data transfer")
dma_recv.transfer(output_buffer)
dma_send.transfer(input_buffer)

print("Buffers")
print(input_buffer)
print(output_buffer)

print("State")
print(dma_send.idle)
print(dma_recv.idle)

#print("Waiting for transfer")
#dma_send.wait()
#dma_recv.wait()

#for i in range(data_size):
#    print(i, input_buffer[i], output_buffer[i])

#del input_buffer
#del output_buffer
#del ol
#print("End of code")
'''