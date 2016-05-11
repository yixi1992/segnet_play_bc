#!/bin/bash

#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="sepf4inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

xixi='sepf1b1f2b2f4b4trgbbs4lr1e-3fixed'
iter_s=16000
iter_e=16000
iter_gap=1000

cur_dir=$PWD
work_dir='/home-4/yixi@umd.edu/segnet/'


module load matlab
for ((n=$iter_s; n<=$iter_e; n+=${iter_gap}))
do
	caffemodel=${cur_dir}/snapshots/${xixi}_iter_${n}.caffemodel
	if [ $n = 0 ];
	then
		caffemodel=${cur_dir}/trainedrgbbs10conv2flow_surg.caffemodel
	fi
	trainprototxt=${cur_dir}/segnet_basic_train.prototxt
	
	rm -r ${cur_dir}/inference/${xixi}_iter_${n}
	mkdir ${cur_dir}/inference/${xixi}_iter_${n}
	python ${work_dir}/compute_bn_statistics_lmdb.py \
		${trainprototxt} \
		${caffemodel} \
		${cur_dir}/inference/${xixi}_iter_${n}/

	rm ${cur_dir}/predictions/inf_${xixi}_iter_${n}/ -r -f
	python /scratch/groups/lsdavis/yixi/segnet/segnetf1/test_segmentation_camvid.py\
		 --model ${cur_dir}/segnet_basic_inference.prototxt\
 		 --weights ${cur_dir}/inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output ${cur_dir}/predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('/scratch/groups/lsdavis/yixi/segnet/segnetf1/compute_test_results'); exit"
done


