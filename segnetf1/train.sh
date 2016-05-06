#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="segnetf1"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000

~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights basic_camvid_surg.caffemodel

