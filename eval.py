import os
os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 

#caffe_root = '../caffe-crfrnn/'
import sys
#sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0,  '../../python')

import numpy as np

import caffe
net = caffe.Net('FCN_deploy.prototxt', 'FCN_iter_120000.caffemodel', caffe.TEST)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
import scipy.io as sio
f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels = f['Label'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Labels1 = f['Label1'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Labels2 = f['Label2'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Labels_AD = f['Label_AD'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1

Conturs = f['Contour'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Conturs1 = f['Contour1'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Conturs2 = f['Contour2'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Conturs_AD = f['Contour_AD'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1

n = 55
Images = Images[n::,:,:512,:]
Labels = Labels[n::,:,:512,:]
Labels1 = Labels1[n::,:,:512,:]
Labels2 = Labels2[n::,:,:512,:]
Labels_AD = Labels_AD[n::,:,:512,:]

Labels_hat = np.zeros((Labels.shape))

#im = Images[0,:,200:328,200:328]
for Idx in range(Images.shape[0]):
    im = Images[Idx,:,:,:]
    im[im==255]   = 10
    in_ = im.reshape(1, *im.shape)#.swapaxes(1,2)
    net.blobs['data'].data[...] = in_
    out = net.forward()['score'][0,:,:,:].swapaxes(0,2)
    Labels_hat[Idx,0,:,:] = out.argmax(axis=2).swapaxes(0,1)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#cm = confusion_matrix( Contour[0,0,:,:].reshape(512*496,),  Contour_hat[0,0,:,:].reshape(512*496,))
imgplot = plt.imshow(Labels[0,0,:,:])
imgplot = plt.imshow(Labels_hat[0,0,:,:])

Labelss = Labels2.reshape(Images.shape[0]*512*496,)
#Labels_hat = Labels
Labels_hat =  Labels_hat.reshape(Images.shape[0]*512*496,)
cm = confusion_matrix( Labelss, Labels_hat)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
import matplotlib.pyplot as plt
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

#imgplot = plt.imshow(out.argmax(axis=2))
#imgplot = plt.imshow(out[:,:,1])
