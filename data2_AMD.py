# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:15:46 2016

@author: root
"""
import os
os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 

import scipy.io as sio
f = sio.loadmat('Data_AMD.mat')
Images = f['Images'].reshape((512,599,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels1 = f['Label1'].reshape((512,599,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Labels2 = f['Label2'].reshape((512,599,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
import numpy as np
Contours1 = f['Contour1'].reshape((512,599,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Contours2 = f['Contour2'].reshape((512,599,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Images = np.concatenate((Images,Images),axis=0)
Labels = np.concatenate((Labels1,Labels2),axis=0)
Contours = np.concatenate((Contours1,Contours2),axis=0)
from scipy import ndimage
n=220

Images = Images[:n,:,:512,:]
Labels = Labels[:n,:,:512,:]
Contours = Contours[:n,:,:512,:]

Images[Images==255] = 10
Labels = Labels[:,0,:,:]
Contours = Contours[:,0,:,:]
from skimage.util.shape import view_as_blocks

#Images  = view_as_blocks(Images , block_shape=(1, 1,128,1)).swapaxes(1,2).reshape(Images.shape[0]*4,3,496,128).swapaxes(2,3)
Images  = view_as_blocks(Images , block_shape=(1, 1,128,1)).reshape(Images.shape[0]*4,1,512,128).swapaxes(2,3)
Labels = view_as_blocks(Labels , block_shape=(1,128,1)).reshape(Labels.shape[0]*4,512,128).swapaxes(1,2)
Contours = view_as_blocks(Contours , block_shape=(1,128,1)).reshape(Contours.shape[0]*4,512,128).swapaxes(1,2)
Contours2 = Contours

for stack in range(Contours.shape[0]):
    Contours2[stack,:,:]  = ndimage.grey_dilation(np.uint(Contours[stack,:,:]), size=(2,2))
#Contours[Contours2==0] =10
Contours=Contours2
from skimage.transform import rescale
Labels_t =  Labels
Labels = Labels/8
Labels_2 = np.zeros((Labels.shape[0],Labels.shape[1]/2,Labels.shape[2]*0.5),dtype='uint8')
for stack in range(Labels.shape[0]):
    Labels_2[stack,:,:] = np.uint(rescale(Labels[stack,:,:],0.5)*8)
Labels_4 = np.zeros((Labels_2.shape[0],Labels_2.shape[1]*0.5,Labels_2.shape[2]*0.5),dtype='uint8')
for stack in range(Labels_2.shape[0]):
    Labels_4[stack,:,:] = np.uint(rescale(Labels_2[stack,:,:],0.5)*8)
Labels_8 = np.zeros((Labels_4.shape[0],Labels_4.shape[1]*0.5,Labels_4.shape[2]*0.5),dtype='uint8')
for stack in range(Labels_4.shape[0]):
    Labels_8[stack,:,:] = np.uint(rescale(Labels_4[stack,:,:],0.5)*8)
Labels_16 = np.zeros((Labels_8.shape[0],Labels_8.shape[1]*0.5,Labels_8.shape[2]*0.5),dtype='uint8')
for stack in range(Labels_4.shape[0]):
    Labels_16[stack,:,:] = np.uint(rescale(Labels_8[stack,:,:],0.5)*8)
Labels=  Labels_t
#Labels = view_as_blocks(Labels , block_shape=(1,1,128,1)).reshape(Labels.shape[0]*4,1,496,128).swapaxes(2,3)
#Contours = view_as_blocks(Contours , block_shape=(1,1,128,1)).reshape(Contours.shape[0]*4,1,496,128).swapaxes(2,3)

#Images = Images[:,:,:,30:374]
#Contour = Contour[:,:,:,30:374]
#CC = CC[:,:,:,30:374]

import h5py
with h5py.File("train_AMD.hdf5", "w") as f:
     dset = f.create_dataset("data", data = Images, dtype='float32')
     dset = f.create_dataset("label", data = Labels, dtype='float32')
     dset = f.create_dataset("label2", data = Contours, dtype='float32')
     dset = f.create_dataset("label_2", data = Labels_2, dtype='float32')
     dset = f.create_dataset("label_4", data = Labels_4, dtype='float32')
     dset = f.create_dataset("label_8", data = Labels_8, dtype='float32')
     dset = f.create_dataset("label_16", data = Labels_16, dtype='float32')
     
#with h5py.File("train_label.hdf5", "w") as f:
#     dset = f.create_dataset("label", data = Contour, dtype='float32')


import matplotlib.pyplot as plt
imgplot = plt.imshow(Images[0,0,:,:])
#imgplot = plt.imshow(Labels[0,0,:,:])
imgplot = plt.imshow(Labels[0,:,:])