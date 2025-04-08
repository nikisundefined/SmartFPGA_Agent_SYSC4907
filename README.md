# Smart FPGA Agent

## Description
This is a capstone project at Carleton University from the 2024-2025 year. The goal is to create a ML agent that can be implemented on a 
PYNQ-Z2 board (ZYNQ 7020 FPGA chipset). The agent is capable of navigating through a video game style level, collecting goals to score
points, with the goals being placed in either random places on the map or being placed directly by the user.

## Features

### Agent
The agent has several features:
- **Navigation**: Uses reinforcement learning to navigate through the environment.
- **Spiking Neurons**: Utilizes spiking neurons for decision-making and signal processing.
- **Prescribed Error Sensitivity**: Allows for adjusting error sensitivity during training through a reward value.
- **Visual Output**: Provides visual feedback through a GUI or console output.
- **Pacman Simulation**: Simulates a Pacman-like game where the agent collects dots (goals).
<p align=center>
<img src=doc/img/local_gui.png alt="Local GUI"></img><br/>
An image of the local gui showing the arena the agent traverses
</p>
<br/>
<p align=center>
<img src=doc/img/nengo_gui.png alt="Nengo GUI"></img><br/>
A image of the nengo gui showing the connections and elements of the neural network
</p>

## Project Structure

```
├── build                                                (Build artifacts)
├── doc
│   └── img                                                (README images)
│       ├── block_design.svg
│       ├── local_gui.png
│       └── nengo_gui.png
├── examples                                (Examples of similar projects)
│   ├── 02-RL-demo.py
│   ├── 02-RL-demo.py.cfg
│   └── grid.py
├── nengofpga                       (Bitstream and Hardware Handoff Files)
│   ├── nengofpga.bit
│   ├── nengofpga.hwh
│   ├── pes_relu_rate.bit
│   ├── pes_relu_rate.hwh
│   ├── pes_relu_spiking.bit
│   └── pes_relu_spiking.hwh
├── nengopy                                       (FPGA wrapper in python)
│   ├── fpga.py
│   ├── __init__.py
│   ├── nengocpp.pyi
│   └── neurons.py
├── smart_agent                                (Simulation and Model Code)
│   ├── gui.py
│   ├── __init__.py
│   ├── model.py
│   ├── model.py.cfg
│   ├── shared.py
│   ├── simulation.py
│   └── vars.py
├── src                                                 (C++ Source Files)
│   ├── bindings                          (Pybind11 bindings for C++ code)
│   │   ├── nengo.cpp
│   │   ├── shared.cpp
│   │   └── simulation.cpp
│   ├── hls                                                 (HLS C++ Code)
│   │   ├── bd.tcl
│   │   ├── rectified_linear.cpp
│   │   └── rectified_linear.h
│   ├── arena.h
│   ├── direction.h
│   ├── nengo.h
│   ├── path_cache.h
│   ├── pathfinding.cpp
│   ├── pathfinding.h
│   ├── path_pair.h
│   ├── player.h
│   ├── point.h
│   ├── shared_arena.h
│   ├── shared_path_cache.h
│   ├── shared_player.h
│   └── shared_point.h
├── apt-packages.txt
├── CMakeLists.txt
├── metrics.json
├── path_cache.json
├── README.md
├── requirements.txt
└── TODO.md
```

## How It Works
### Simulation
The simulation is written completely in python and composed of an:
- **Arena**: This is a 2D grid representing the game level. It is also composed of the player and the goal
- **Player**: This is the agent that navigates through the arena. It has a position and can move in four directions.
- **PlayerInfo** & **Performance**: These classes track the agent's performance over time.
- **PathCache**: This class caches paths for efficient navigation.

The general flow of the simulation is:
1. An Arena instance is started which contains the player.
2. The agent moves within the bounds of the arena with the `move()` method
3. A check is performed to see if the player is currently on a goal
4. If so, a new goal is selected

### Agent
The agent is a simple reinforcement learning model that learns to navigate through the arena.
It uses a reward generated by its actions to scale the error.
Error of the model is calculated using PES (Prescribed Error Sensitivity)
The expected output of the model is a direction to move in. The error is calculated as the difference between the expected output and the actual output.

#### Error
Error is determined as follows:
1. The best direction the agent can move in is computed. The expected value for this direction is set to <a href=smart_agent/model.py#L203>`BASELINE_ERROR`</a>
2. The expected values of all other directions are set to 0
3. The difference between the current value and the expected values is taken
4. The error scaling value is calculated and scales the baseline error

Error Scaling is based on the reward of the agent using the following equation:

<p align=center>
  
$`
f(x) =
\begin{cases}
\frac{12}{Reward} + 0.0.1 & \text{if } x \ge 50 \\
\frac{\sqrt{-(Reward - 50)}}{8} + 0.25 & \text{if } 1 \le x < 50
\end{cases}
`$

</p>

#### Reward
The reward for the agent is based on the following actions:
- +1: Any Movement
- -5: Stopping
- +2: Move towards Goal
- -2: Move away from Goal
- -3: Last 2 movements moved away from goal
- -5: Hit a wall
- -2: Repeated position (returned to the same tile at least 2 times, in the last 5 actions)
- -4: Many repeated positions (returned to the same tile at least 4 times, in the last 5 actions)

### Learning 
Learning of the model is done through nengo using nengo's builtin learning rule.
For more information about how the learning rule works look <a href=https://www.nengo.ai/nengo/frontend-api.html#nengo.PES>here</a>.

### FPGA
The FPGA component is implenented by subclassing nengos <a>`RectifiedLinear`</a> neuron type and is split into 3 sections.

#### Python
A Python wrapper is included to translate calls for functions from Python into a data structure useable by the HLS ip in the FPGA.
The wrapper also translates the return value from the FPGA into the expected output of the Python function call (ex. returning tuples of numpy array).
More details about the strutcure of the HLS function and its inputs/outputs are given in the following sections.

#### HLS
The top level function of the IP accepts a 256-bit vector as both an input and output.
This vector is split into 4 64-bit components:
1. The ID of the function to call (defined in <a href=src/hls/rectified_linear.cpp#L111>`rectified_linear.cpp`</a>)
2. The first parameter of the function (converted to double)
3. The second parameter of the function (converted to double)
4. The third parameter of the function (converted to double)

The function then calls the appropriate C++ function based on the ID of the function passed in.
Output products are then written back into the output vector in the same order as the inputs.

#### Vivado

<p align=center>
<img src=doc/img/block_design.svg></img><br/>
Image of block design used in Vivado
</p>

There are 2 main components involved in the block design shown above.
1. **AXI DMA**: This block reads/writes data from/to memory and passes information to HLS IP as an AXI FIFO stream.
2. **HLS IP Core**: This core contains the HLS function that was described in the previous section.

An important note about the AXI DMA is that if a channel is busy/waiting for information, the entire channel will be blocked until enough information is passed through the stream. For this reason, it is recommended to only read/write to the same buffer as the input and output sizes are the same, there is no chance for the DMA transfers to stall waiting for read/write capacity. For more information about AXI DMA and other things to watch out for refer to <a href=nengopy/fpga.py#L23>`fpga.py`</a>

## Installation

### Requirements
>This project requires:
>- python >= 3.10
>- numpy <= 1.26.2
>- pynq (if running on FPGA else optional)
>- nengo
>- nengo-spa>- nengo
>- nengo-spa
>- nengo-gui
>- nengo-gui
>- dearpygui (optional, used for local gui)

First clone this repository and navigate to the project directory. Then, create a virtual environment and activate it:
>If you are running on the PYNQ Linux image you can skip this step.
```bash
$ git clone https://github.com/nikisundefined/SmartFPGA_Agent_SYSC4907.git
$ cd SmartFPGA_Agent_SYSC4907
$ python3 -m venv venv
$ source venv/bin/activate
```

If you want to build the HLS IP and bitstream files, you will need to install the Xilinx Vivado and Vitis HLS.
Follow the next section about building the bistream for the board:

> ## Note:
> It is not necessary to build the Vivado and Vitis projects as the bitstream and hardware handoff are not locked to any board and are included in this repository as <a href=nengofpga/nengofpga.bit>`nengofpga.bit`</a> and <a href=nengofpga/nengofpga.hwh>`nengofpga.hwh`</a>

### Building

### HLS
1. In Vitis create a new project targeting the Zynq-7020 or xc7z020-clg400-1, If you use another board or FPGA core, choose that
2. Add the source files <a href=src/hls/rectified_linear.h>`rectified_linear.h`</a> and <a href=src/hls/rectified_linear.cpp>`rectified_linear.cpp`</a> as the source files
3. Select the top level function as `nengofpga` and skip everything else
4. Perform the C Synthesis and Export RTL

You should now be able to import the IP into Vivado for the next step.
### Vivado
1. Open Vivado and create a new project targeting the Zynq-7020 or xc7z020-clg400-1, If you use another board or FPGA core, choose that
2. Do not add any source files to the project and continue through all dialogs until you finish the setup wizard
3. Open the IP catalog and add the folder where you created the HLS project, you should now see the nengofpga IP component available under the custom repository
4. Source the TCL file: <a href=src/hls/bd.tcl>`bd.tcl`</a> in the console, the block should be generated for you
5. Generate the bistream and hit okay for any dialogs that open
6. Run `File > Export > Export Bitstream...` and save the file as `nengofpga.bit`
7. Find the file `design_1.hwh` under the following path: `<Vivado Project Folder>/<Vivado Project Name>.gen/sources_1/bd/design_1/hw_handoff`

You now have the necessary files to run the project on the FPGA
> ### IMPORTANT:
> Ensure the `.bit` and `.hwh` file have the same name otherwise pynq will not recognize them. The name is meaningless to the actual bitstream and only used by pynq to correlate files.

### Setup

Ensure the required packages are installed, run the following command:
```bash
$ sudo apt-get update && sudo apt-get install -y $(cat apt-packages.txt)
```

Next, install the Python dependencies using pip:
```bash
(venv) $ pip install -r requirements.txt
```

## Usage

To start the agent with the nengo GUI for internal graphing and debugging, run:
```bash
(venv) $ python -m smart_agent.model --nengo
```

This will automatically launch the nengo GUI as well as the local GUI in a separate thread.
> ### Note:
> If running the model with the FPGA component, it is essential to run with root privileges. This can be done by prefixing the command with `sudo` or adding your user to the appropriate group. ex:
> ```bash
> $ sudo -E python -m smart_agent.model --nengo
> ```
