#!/bin/bash

#SBATCH --job-name=prefixPara
#SBATCH --gres=gpu:0
#SBATCH --partition=teach_gpu
#SBATCH --account=COMS031424
#SBATCH --output=lbm.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1


echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`
echo GPU number: $CUDA_VISIBLE_DEVICES

export OCL_DEVICE=1

module load libs/cuda/10.0-gcc-5.4.0-2.26
module use /software/x86/tools/nvidia/hpc_sdk/modulefiles
module load NVIDIA/nvhpc/21.9

#! Run the executable
#./d2q9-bgk input_128x128.params obstacles_128x128.dat
#./d2q9-bgk input_256x256.params obstacles_256x256.dat
#./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
nvprof -o my_profile_report ./parallel_scan

