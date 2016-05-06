import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from PIL import Image
from sklearn.preprocessing import normalize
caffe_root = '/scratch/groups/lsdavis/yixi/software/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

outputs = args.output
if not os.path.exists(outputs):
	os.makedirs(outputs)

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

idx=0
acc=[]
softmaxlosses = np.array([])
accuracys = np.array([])
batch_size = net.blobs['data'].num
print 'batch_size=' + str(batch_size)
for i in range(0, int(np.ceil(args.iter/float(batch_size)))):
		
	net.forward()
	#print i
	#for ii in range(0, len(net.params['conv1_flow'])):
	#	print ii, np.sum(net.params['conv1_flow'][ii].data**2)
	
	image0 = net.blobs['data'].data
	label0 = net.blobs['label'].data
	predicted0 = net.blobs['prob'].data
	for j in range(net.blobs['data'].num): 
		image = np.squeeze(image0[j,:,:,:])
		label = np.array(np.squeeze(label0[j,:,:,:]), dtype=np.uint8)
		output = np.squeeze(predicted0[j,:,:,:])
		ind = np.argmax(output, axis=0)
		ind = np.array(ind, dtype=np.uint8)

		#ind = np.array(np.squeeze(net.blobs['argmax'].data), dtype=np.uint8)	

		softmaxloss = np.array(np.squeeze(net.blobs['loss'].data))
		softmaxlosses = np.append(softmaxlosses, softmaxloss)
		print 'softmaxloss=',softmaxloss,' mean=', np.mean(softmaxlosses)
	
		accuracy = np.array(np.squeeze(net.blobs['accuracy'].data))
		accuracys = np.append(accuracys, accuracy)
		print 'accuracy=', accuracy, 'mean=', np.mean(accuracys)
		def findsource(image, label):
			input_source='/scratch/groups/lsdavis/yixi/segnet/CamVid/test.txt'
			for line in open(input_source,'r'):
				files = line.rstrip('\n').split(' ')
				im_file = files[0]
				annot_file = files[1]

				gt_l = np.array(Image.open(annot_file))

				im = np.array(Image.open(im_file))
				im = im[:,:,::-1]
				im = im.transpose((2,0,1))
				
				image_dis = np.sum((image[:3,:,:]-im)**2)
				label_dis = np.sum((label-gt_l)**2)
				print 'data',i,'file',im_file, 'image_dis=',image_dis, 'label_dis=',label_dis	
				if image_dis<1e-6 and label_dis<1e-6:
					return im_file
			return -1	



		#o = findsource(image, label)
		#if o==-1:
	#		print 'err cannot find original file for data ',i
	#		exit(1)
	#	else:
	#		print 'success matched data ', i, 'with', o
		IMAGE_FILE = os.path.join(outputs, str(idx))
		idx = idx+1		
		scipy.misc.imsave(IMAGE_FILE+'_pr.png', ind)
		scipy.misc.imsave(IMAGE_FILE+'_gt.png', label)
		if idx>=args.iter:
			exit(0)	
	
	if False:	
		label_flat = label
		print np.sum(np.equal(ind,label_flat))
		
		acc = acc+[float(np.sum(np.equal(ind,label_flat)))/np.sum(label_flat!=11)]
		print 'acc=', np.sum(np.equal(ind,label_flat)), '/', np.sum(label_flat!=11), '=', float(np.sum(np.equal(ind,label_flat)))/np.sum(label_flat!=11)
		#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')


		r = ind.copy()
		g = ind.copy()
		b = ind.copy()
		r_gt = label.copy()
		g_gt = label.copy()
		b_gt = label.copy()

		Sky = [128,128,128]
		Building = [128,0,0]
		Pole = [192,192,128]
		Road_marking = [255,69,0]
		Road = [128,64,128]
		Pavement = [60,40,222]
		Tree = [128,128,0]
		SignSymbol = [192,128,128]
		Fence = [64,64,128]
		Car = [64,0,128]
		Pedestrian = [64,64,0]
		Bicyclist = [0,128,192]
		Unlabelled = [0,0,0]

		label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
		for l in range(0,11):
			r[ind==l] = label_colours[l,0]
			g[ind==l] = label_colours[l,1]
			b[ind==l] = label_colours[l,2]
			r_gt[label==l] = label_colours[l,0]
			g_gt[label==l] = label_colours[l,1]
			b_gt[label==l] = label_colours[l,2]

		rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb[:,:,0] = r/255.0
		rgb[:,:,1] = g/255.0
		rgb[:,:,2] = b/255.0
		rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb_gt[:,:,0] = r_gt/255.0
		rgb_gt[:,:,1] = g_gt/255.0
		rgb_gt[:,:,2] = b_gt/255.0

		image = image/255.0

		image = np.transpose(image, (1,2,0))
		image = image[:,:,(2,1,0)]

		fig = plt.figure()
		plt.imshow(image)
		scipy.misc.imsave(IMAGE_FILE+'_image.png', image)
		plt.close(fig)
		
		fig = plt.figure()
		plt.imshow(rgb_gt)
		scipy.misc.imsave(IMAGE_FILE+'_rgbgt.png', rgb_gt)
		plt.close(fig)

		fig = plt.figure()
		plt.imshow(rgb)
		scipy.misc.imsave(IMAGE_FILE+'_rgbpr.png', rgb)
		plt.close(fig)
		#plt.show()

		if False:
			plt.figure()
			plt.imshow(image,vmin=0, vmax=1)
			plt.figure()
			plt.imshow(rgb_gt,vmin=0, vmax=1)
			plt.figure()
			plt.imshow(rgb,vmin=0, vmax=1)
			plt.show()


print 'Success!'
print 'acc=', np.mean(acc)
