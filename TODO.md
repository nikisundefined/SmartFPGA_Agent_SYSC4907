Notes:
    Consider Super/Sub reward state (Super state = Score * Time, Sub state = distance to goal)
    Scale Super/Sub reward state error differently (Super state = scale global error, Sub state = scale directional error)
    Consider increasing the error for the second best move in the path
    Conversion of nengo neuron model to C++ HLS equivalent
        - Convert python implementation to C++
        - Create python proxy class to wrap C++ implementation using something like pybind11
        - Replace implementation of neuron model in library with custom model

TODO:
    RIGHTNOW:
        - [x] Clean up metrics implementation
        - [x] JSON export of Performance
        - [x] Add goal locations to Performance
        - [ ] Test running model on board
        - [x] Fix cache implementation
            - [x] When generating a path add all sub paths (0 -> n, 1 -> n, 2 -> n) to the cache to prevent recompute along the current path
        - [ ] Test local GUI on board
        - [ ] Integrate path caching in C++
        - [ ] Fix main code to run without nengo gui
    Reduce memory usage
        - [x] C++ implementation
            - [ ] Integrate into `model.py`
            - [ ] Rewrite `simulation.py` and `shared.py` to import C++ implementation
            - [ ] Test C++ implementation 
    Finish implementation of NeuronTypes in C++/HLS
        - [ ] NeuronType
            - [ ] current
            - [ ] gain_bias
            - [ ] max_rates_intercepts
            - [ ] rates
            - [ ] step
        - [x] RectifiedLinear
            - [x] gain_bias
            - [x] max_rates_intercepts
            - [x] step
        - [ ] SpikingRectifiedLinear
            - [ ] rates
            - [x] step
        - [ ] LIF
            - [ ] LIFRate
                - [ ] gain_bias
                - [ ] max_rates_intercepts
                - [ ] rates
                - [ ] step
            - [ ] step
        - [ ] AdaptiveLIF
            - [ ] AdaptiveLIFRate
                - [ ] step
            - [ ] step
    Update Requirements and make setup scripts
        - [ ] Conda requirements
        - [ ] Pip requirements
        - [ ] C++ requirements
    Add Vivado and Vitis Project to Git
        - [ ] Vitis HLS
        - [ ] Vivado Block Design
    Test other inputs
        - [ ] Player Position
        - [ ] Goal Position
        - [x] Detection Distance
    Check if the model is actually learning or just adapting based on the error
        - Seems to only be adapting based on the error
        - [ ] Add noise to inputs/outputs/error
    Add more tracking in `Player` class
        - [ ] Steps to reach goal
        - [ ] Reward Value
        - [ ] Time Taken (Real Time)
    Add performance characteristics
        - [ ] Time per goal
            - Change where it is updated to only update on whole number
        - [ ] Movements per goal
            - Change where it is updated so its only updated on move
        - [x] Reward Value at goal
        - [ ] Amount of goal reached in x seconds
    Generate list of hyperparameters for optimization phase
        - Learning Rate
        - Error Baseline
        - Reward Factors
        - Neuron Count