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


#--weights snapshots/${xixi}_iter_${n}.caffemodel\
xixi='lr0.1'
for n in {9000..9000..1000}
do
	rm predictions/${xixi}_iter_${n}/ -r -f
	python test_segmentation_camvid.py\
		 --model segnet_basic_inference.prototxt\
		--weights inference/test_weights.caffemodel\
		 --iter 233\
		 --output predictions/${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = 'predictions/${xixi}_iter_${n}/*_gt.png'; predPath = 'predictions/${xixi}_iter_${n}/*_pr.png'; run('compute_test_results'); exit"
done
END
