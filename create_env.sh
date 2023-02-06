#!/bin/bash
# THIS SCRIPT HAS TO BE EXECUTED FROM PYRIL DIRECTORY
export CONDA_ALWAYS_YES="true"

# Create conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate vsn

# Install pytorch and clip dependencies
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/openai/CLIP.git

unset CONDA_ALWAYS_YES