#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="flowonly"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000



#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -snapshot snapshots/f1bs10lr1e-1fixed_iter_3000.solverstate
~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -snapshot snapshots/flowonlybs10lr1e-3fixed_iter_10000.solverstate
