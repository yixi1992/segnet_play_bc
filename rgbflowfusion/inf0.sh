#!/bin/bash

#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="rgbflowfusion"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

cur_dir=$PWD
work_dir='/home-4/yixi@umd.edu/segnet/'
rgb_trainprototxt='/scratch/groups/lsdavis/yixi/segnet/repcamvid/segnet_basic_train.prototxt'
rgb_caffemodel='/scratch/groups/lsdavis/yixi/segnet/repcamvid/snapshots/bs10lr0.1_iter_4200.caffemodel'

flow_trainprototxt=${cur_dir}/segnet_basic_train.prototxt

rm -r ${cur_dir}/inference/rgb
mkdir ${cur_dir}/inference/rgb
python ${work_dir}/compute_bn_statistics_all.py \
		${rgb_trainprototxt} \
		${rgb_caffemodel} \
		${cur_dir}/inference/rgb/

	inferenceprototxt=${cur_dir}/segnet_basic_inference.prototxt
	if [ "$fidl" = true ];
	then
		inferenceprototxt=${cur_dir}/segnet_basic_inference_slice_fidl.prototxt
	fi

	rm ${cur_dir}/predictions/inf_${xixi}_iter_${n}/ -r -f
	python /scratch/groups/lsdavis/yixi/segnet/segnetf1/test_segmentation_camvid.py\
		 --model ${inferenceprototxt}\
 		 --weights ${cur_dir}/inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output ${cur_dir}/predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('/scratch/groups/lsdavis/yixi/segnet/segnetf1/compute_test_results'); exit"
done


