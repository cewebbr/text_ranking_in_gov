#!/usr/bin/env bash

# More info at: https://www.tensorflow.org/install/pip

# Settings:
envname=icedeg

# Create and enter environment:
conda create --name $envname python=3.9
eval "$(conda shell.bash hook)"
conda activate $envname

# For GPU usage:
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

# Set paths:
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Maybe reloading the environment is required to access the path above:
conda deactivate
conda activate $envname

# Install packages:
pip install -r requirements.txt
