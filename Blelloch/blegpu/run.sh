#!/bin/bash
#PBS --job-name=prefixPara
#PBS --account=br-baram
#PBS --partition=gpu
#PBS--nodes=1
#PBS --ntasks-per-node=1
#PBS --cpus-per-task=4
#PBS --time=0:01:00
#PBS --gres=gpu:1
#
module load cuda11.2/toolkit/11.2.0
##

nvcc -c Submission.cu

# Link your code
nvcc Submission.o -o myProgram  # Replace myProgram with your program name

# Run your program
./myProgram
