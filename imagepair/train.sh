#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="sepf1"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

~/work/yixi/software/caffe-segnet-yixi/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights trainedrgbbs10conv2flow_surg.caffemodel

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/sepf1/snapshots/trgbbs4lr1e-3fixed_iter_19000.solverstate
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights trainedrgbbs10conv2flow_surg.caffemodel
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights trainedrgbbs10conv2flow_msrainit_surg.caffemodel
