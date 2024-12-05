#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
module purge
module load Python/3.10

# Create a virtual environment
# python3 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/pmnsbandara/myenv/bin/activate

# Packages
pip install traceback
pip install matplotlib
pip install numpy
pip install pillow
pip install networkx
pip install scipy
pip install scikit-learn
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html --no-cache-dir
pip install torch-scatter
pip install Bio
pip install transformers
pip install torchmetrics
pip install biopandas
pip install periodictable
pip install pandas

srun whichgpu

srun --gres=gpu:1 python ~/P2PXML_Structure/integrated_model_v1.py
