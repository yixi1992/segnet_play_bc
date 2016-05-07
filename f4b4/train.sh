#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f4b4"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000


#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/f1b1/segnet_basic_solver.prototxt -weights ~/segnet/f1b1/trainedf1bs10_surg.caffemodel

~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights trainedrgbbs10_surg.caffemodel

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/f1b1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/f1b1/snapshots/f1b1trgbbs10lr1e-3fixed_iter_6000.solverstate
