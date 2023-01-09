#!/bin/bash

#SBATCH --job-name="MLP_MGPU"
#SBATCH --qos=training
#SBATCH -D .
#SBATCH --output=LAB3/%x_%j.out
#SBATCH --error=LAB3/%x_%j.err
#SBATCH --cpus-per-task=160
#SBATCH --gres gpu:4
#SBATCH --time=00:02:00

module purge; module load ffmpeg/4.0.2 gcc/6.4.0 cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 opencv/3.4.1 python/3.6.5_ML 
# module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML tensorflow/2.6.0

# python LAB3/gradient_descent.py
# python LAB3/singlelayer.py
# python LAB3/multilayer.py
python LAB3/multigpu.py
