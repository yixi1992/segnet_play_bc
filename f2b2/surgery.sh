#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=END
#SBATCH --job-name="surgery"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

:<<'END'
python ~/segnet/surgery_flow.py \
	-f /scratch/groups/lsdavis/yixi/segnet/segnetf1/segnet_basic_train_batchsize.prototxt \
	-c /home-4/yixi@umd.edu/segnet/snapshots/trgbbs10learngglr1e-1fixed_iter_900.caffemodel \
	-t /home-4/yixi@umd.edu/segnet/f1b1/segnet_basic_train_batchsize_defaultconv1.prototxt \
	-o /home-4/yixi@umd.edu/segnet/f1b1/trainedf1bs10_surg.caffemodel \
	--fromlayer='conv1_flow' \
	--tolayer='conv1_f1b1' \
END

python ~/segnet/surgery_flow.py \
	-f /scratch/groups/lsdavis/yixi/segnet/repcamvid/segnet_basic_train.prototxt \
	-c /scratch/groups/lsdavis/yixi/segnet/repcamvid/snapshots/bs10lr0.1_iter_4200.caffemodel \
	-t segnet_basic_train_defaultconv1.prototxt \
	-o trainedrgbbs10_surg.caffemodel \
	--fromlayer='conv1' \
	--tolayer='conv1_f1b1f2b2' \

