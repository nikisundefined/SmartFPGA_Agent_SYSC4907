#!/bin/bash

# Ensure nengo environment exists otherwise create it
env_exists=$(conda info --envs)
if [[ "$env_exists" != *"nengo"* ]]; then
    conda create nengo
fi

conda deactivate
# Activate the environment
conda activate nengo

# Setup dependencies
conda install python==3.10.0 nengo==4.0.0 nengo-extras nengo-spa nengo-gui nengo-ocl
pip install -r requirements.txt