#!/bin/bash

#SBATCH -t 3:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f1inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

xixi='f1b1trgbbs10lr1e-3fixed'
bs=true
slice=false
fromrgb=true
iter_s=7000
iter_e=8000
iter_gap=1000

cur_dir='/home-4/yixi@umd.edu/segnet/f1b1/'
work_dir='/scratch/groups/lsdavis/yixi/segnet/segnetf1/'

cd ${work_dir}


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
	if [ "$bs" = true ];
	then
		trainprototxt=${cur_dir}/segnet_basic_train_batchsize.prototxt
	fi
	if [ "$slice" = true ];
	then
		trainprototxt=${cur_dir}/segnet_basic_train_slice.prototxt
	fi
	
	inferenceprototxt=${cur_dir}/segnet_basic_inference.prototxt
	if [ "$slice" = true ];
	then
		inferenceprototxt=${cur_dir}/segnet_basic_inference_slice.prototxt
	fi
	echo $bs
	echo $slice
	echo $caffemodel
	echo $trainprototxt
	echo $inferenceprototxt

	rm -r ${cur_dir}/inference/${xixi}_iter_${n}
	mkdir ${cur_dir}/inference/${xixi}_iter_${n}
	python ${cur_dir}/../compute_bn_statistics_lmdb.py \
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


