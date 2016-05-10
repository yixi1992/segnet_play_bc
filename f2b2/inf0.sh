#!/bin/bash

#SBATCH -t 3:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f2b2inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

xixi='f1b1f2b2trgbslicefidlbs4lr1e-3fixed'
bs_bn=true
slice=true
fromrgb=true
fidl=true
iter_s=0
iter_e=30000
iter_gap=1000

cur_dir=$PWD
work_dir='/scratch/groups/lsdavis/yixi/segnet/segnetf1/'



module load matlab
for ((n=$iter_s; n<=$iter_e; n+=${iter_gap}))
do
	caffemodel=${cur_dir}/snapshots/${xixi}_iter_${n}.caffemodel
	if [ $n = 0 ];
	then
		caffemodel=${cur_dir}/trainedf1bs10_surg.caffemodel
		if [ "$fromrgb" = true ];
		then
			caffemodel=${cur_dir}/trainedrgbbs10_surg.caffemodel
		fi
	fi
	
	trainprototxt=${cur_dir}/segnet_basic_train.prototxt
	if [ "$bs_bn" = true ];
	then
		trainprototxt=${cur_dir}/segnet_basic_train_batchsize.prototxt
		if [ "$slice" = true ];
		then
			trainprototxt=${cur_dir}/segnet_basic_train_slice_batchsize.prototxt
		fi
	else
		if [ "$slice" = true ];
		then
			trainprototxt=${cur_dir}/segnet_basic_train_slice.prototxt
			if [ "$fidl" = true ];
			then
				trainprototxt=${cur_dir}/segnet_basic_train_slice_fidl.prototxt
			fi
		fi
	fi
	
	inferenceprototxt=${cur_dir}/segnet_basic_inference.prototxt
	if [ "$slice" = true ];
	then
		inferenceprototxt=${cur_dir}/segnet_basic_inference_slice.prototxt
		if [ "$fidl" = true ];
		then
			inferenceprototxt=${cur_dir}/segnet_basic_inference_slice_fidl.prototxt
		fi
	fi
	echo $bs_bn
	echo $slice
	echo $caffemodel
	echo $trainprototxt
	echo $inferenceprototxt

	rm -r ${cur_dir}/inference/${xixi}_iter_${n}
	mkdir ${cur_dir}/inference/${xixi}_iter_${n}
	python /home-4/yixi@umd.edu/segnet/compute_bn_statistics_lmdb.py \
		${trainprototxt} \
		${caffemodel} \
		${cur_dir}/inference/${xixi}_iter_${n}/

	rm ${cur_dir}/predictions/inf_${xixi}_iter_${n}/ -r -f
	python ${work_dir}/test_segmentation_camvid.py\
		 --model ${inferenceprototxt}\
 		 --weights ${cur_dir}/inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output ${cur_dir}/predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('${work_dir}/compute_test_results'); exit"
done



