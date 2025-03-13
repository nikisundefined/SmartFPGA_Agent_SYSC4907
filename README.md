# Smart FPGA Agent
This is a capstone project at Carleton University from the 2024-2025 year. The goal is to create a ML agent that can be implemented on a 
PYNQ-Z2 board (ZYNQ 7020 FPGA chipset). The agent is capable of navigating through a video game style level, collecting goals to score
points, with the goals being placed in either random places on the map or being placed directly by the user.

## Agent
The agent is programmed using the Nengo library to create the model on the software side, with Vivado/Vitis being used to create the
hardware implementation. The hardware implementation is neurons that are communicated to via DMA provided by the PYNQ library.
