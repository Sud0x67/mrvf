#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.7 -y
# conda activate pymarl

conda install pytorch torchvision cudatoolkit=11.0 -c pytorch -y
pip install -r requirmetnts.txt
pip install git+https://github.com/oxwhirl/smac.git
