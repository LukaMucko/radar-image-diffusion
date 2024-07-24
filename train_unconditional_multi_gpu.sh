#!/bin/bash

#PBS -N unconditional_multi_gpu
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=2:mem=20GB
#PBS -o ~/diffusion/out/
#PBS -e ~/diffusion/out/

module load cray-python
cd ~/diffusion/
accelerate launch --multi_gpu train_unconditional.py --train_data_dir images/ --resolution 256 --dataloader_num_workers 4 --output_dir model/