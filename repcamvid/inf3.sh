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

#python test_segmentation_basic_camvid.py\
# --model ../Example_Models/segnet_basic_camvid.prototxt\
# --weights ../Models/Inference/segnet_basic_camvid.caffemodel\
# --input /scratch/groups/lsdavis/yixi/segnet/CamVid/test.txt\
# --output predictions/basic/

 #--weights snapshots_whole/gglr1e-4fixedadagrad_iter_100.caffemodel\
#snapshots_whole/ftrgbgglr1e-4fixed_iter_100.caffemodel

xixi='bs10lr0.1'
iter=4900

mkdir ${xixi}_iter_${iter}

python compute_bn_statistics.py segnet_basic_train.prototxt snapshots/${xixi}_iter_${iter}.caffemodel inference/${xixi}_iter_${iter}/


module load matlab
: <<'END'
n=0
python test_segmentation_camvid.py\
		 --model ftrgb_inference.prototxt\
		 --weights ../Models/Inference/segnet_basic_camvid.caffemodel \
		 --iter 233\
		 --output predictions/ftrgbgglr1e-4fixed_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = 'predictions/ftrgbgglr1e-4fixed_iter_${n}/*_gt.png'; predPath = 'predictions/ftrgbgglr1e-4fixed_iter_${n}/*_pr.png'; run('compute_test_results'); exit"
END


for ((n=$iter; n<=$iter; n+=100))
do
	rm predictions/inf_${xixi}_iter_${n}/ -r -f
	python test_segmentation_camvid.py\
		 --model segnet_basic_inference2.prototxt\
 		--weights inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = 'predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = 'predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('compute_test_results'); exit"
done
