#!/bin/bash

#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f1inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000


python test_segmentation_camvid.py\
 --model segnet_basic_inference.prototxt\
 --weights snapshots_whole/lr1e-6_iter_100.caffemodel\
 --iter 233\
 --output predictions/lr1e-6_iter_100/

