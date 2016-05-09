import os, glob
import numpy as np
from PIL import Image
from collections import namedtuple
from random import shuffle
import shutil


if __name__=='__main__':
	if True:
		print 'convert to lmdb begins....\n'
		resize = True
		RSize = (480, 360)
		LabelSize = (480, 360)
		nopadding = True
		use_flow = ['f1','b1', 'f2', 'b2', 'f4', 'b4']
		flow_dirs = ['flow_x', 'flow_y']
		RGB_mean_pad = False
		flow_mean_pad = True
		# Default is RGB_mean_pad = False and flow_mean_pad = True
		
		RGB_pad_values = [] if RGB_mean_pad else [0,0,0]
		flow_pad_value = 128 if flow_mean_pad else 0


		lmdb_dir = 'camvidpiletrainval' + ('rgbmp' if RGB_mean_pad else '') + ('fmp' if flow_mean_pad else '') + str(RSize[0]) + str(RSize[1]) + (''.join(use_flow)) + ('np' if nopadding else '') + '_txts'
		
		class CArgs(object):
			pass	
		args = CArgs()
		args.resize = resize
		args.RSize = RSize
		args.LabelSize = LabelSize
		args.nopadding = nopadding
		args.use_flow = use_flow
		args.RGB_mean_pad = RGB_mean_pad
		args.flow_mean_pad =flow_mean_pad
		args.RGB_pad_values = RGB_pad_values
		args.flow_pad_value = flow_pad_value
		args.BoxSize = None # None is padding to the square of the longer edge
		args.NumLabels = 12 # [0,11]
		args.BackGroundLabel = 11
		args.lmdb_dir = lmdb_dir
		#args.proc_rank = proc_rank
		#args.proc_size = proc_size		

		#train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
		train_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/train/{id}.png'
		val_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/val/{id}.png'
		test_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/test/{id}.png'
	 	train_label_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/trainannot/{id}.png'
	 	val_label_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/valannot/{id}.png'
	 	test_label_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/testannot/{id}.png'
	 	flow_data = '/scratch/groups/lsdavis/yixi/segnet/CamVid/flow/{id}.{flow_type}.{flow_dir}.png'
		train_keys = [line.rstrip('\n') for line in open('/scratch/groups/lsdavis/yixi/segnet/CamVid/p_train.txt')]
		val_keys = [line.rstrip('\n') for line in open('/scratch/groups/lsdavis/yixi/segnet/CamVid/p_val.txt')]
		test_keys = [line.rstrip('\n') for line in open('/scratch/groups/lsdavis/yixi/segnet/CamVid/p_test.txt')]

	
		inputs_all = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
		
		inputs_Train = dict([(k, train_data.format(id=k)) for k in train_keys])
		inputs_Val = dict([(k, val_data.format(id=k)) for k in val_keys])
		inputs_Test = dict([(k, test_data.format(id=k)) for k in test_keys])
		
		inputs_Train_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Train.keys()])	
		inputs_Val_Label = dict([(id, val_label_data.format(id=id)) for id in inputs_Val.keys()])
		inputs_Test_Label = dict([(id, test_label_data.format(id=id)) for id in inputs_Test.keys()])
		
		Train_keys = inputs_Train.keys()
		shuffle(Train_keys)
		Val_keys = inputs_Val.keys()
		shuffle(Val_keys)
		Test_keys = inputs_Test.keys()
		shuffle(Test_keys)

		flow_Train = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Train_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Val = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Val_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Test = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Test_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 


		inputs_TrainVal = inputs_Train.copy()
		inputs_TrainVal.update(inputs_Val)
		
		inputs_TrainVal_Label = inputs_Train_Label.copy()
		inputs_TrainVal_Label.update(inputs_Val_Label)
		
		flow_TrainVal = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Train_keys] + [(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Val_keys])  for flow_dir in flow_dirs for flow_type in use_flow] 
		
		TrainVal_keys = inputs_TrainVal.keys()
		shuffle(TrainVal_keys)


		if os.path.exists(lmdb_dir):
			shutil.rmtree(lmdb_dir, ignore_errors=True)

		if not os.path.exists(lmdb_dir):
			os.makedirs(lmdb_dir)

		def outputToText(Train, filepath, Train_keys):
			f = open(filepath, 'w')
			for key in sorted(Train_keys):
				for arr in Train:
					f.write(arr[key]+' ');
				f.write("\n")
			f.close()
			
		outputToText([inputs_Train]+[inputs_Train_Label]+flow_Train, os.path.join(lmdb_dir, 'train.txt'), Train_keys)	
		outputToText([inputs_Val]+[inputs_Val_Label]+flow_Val, os.path.join(lmdb_dir, 'val.txt'), Val_keys)
		outputToText([inputs_Test]+[inputs_Test_Label]+flow_Test, os.path.join(lmdb_dir, 'test.txt'), Test_keys)
