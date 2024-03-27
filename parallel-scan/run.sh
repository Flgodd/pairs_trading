#!/bin/bash
#SBATCH --job-name=prefixPara
#SBATCH --account=br-baram
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:01:00
#SBATCH --gres=gpu:1
#
module load cuda11.2/toolkit/11.2.0
##

nvcc -c Submission.cu

# Link your code
nvcc Submission.o -o myProgram  # Replace myProgram with your program name

# Run your program
./myProgram
