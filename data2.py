# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:15:46 2016

@author: root
"""
import os
#os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 

import scipy.io as sio
f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Contour = f['Label'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
import numpy as np
CC = f['Contour'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
n=110
Images = Images[:n,:,:512,:]
Contour = Contour[:n,:,:512,:]
CC = CC[:n,:,:512,:]
from scipy import ndimage
for stack in range(CC.shape[0]):
    CC[stack,0,:,:] = ndimage.grey_dilation(np.uint(CC[stack,0,:,:]), size=(2,2))

Images[Images==255] = 10
Contour = Contour[:,:,:,:]
CC = CC[:,:,:,:]
from skimage.util.shape import view_as_blocks


import h5py
with h5py.File("train.hdf5", "w") as f:
     dset = f.create_dataset("data", data = Images, dtype='float32')
     dset = f.create_dataset("label", data = Contour, dtype='float32')
     dset = f.create_dataset("label2", data = CC, dtype='float32')
f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Contour = f['Label'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
import numpy as np
CC = f['Contour'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Images = Images[n::,:,:512,:]
Contour = Contour[n::,:,:512,:]
CC = CC[n::,:,:512,:]
from scipy import ndimage

for stack in range(CC.shape[0]):
    CC[stack,0,:,:] = ndimage.grey_dilation(np.uint(CC[stack,0,:,:]), size=(2,2))

Images[Images==255] = 10
Contour = Contour[:,:,:,:]
CC = CC[:,:,:,:]
from skimage.util.shape import view_as_blocks


import h5py
with h5py.File("test.hdf5", "w") as f:
     dset = f.create_dataset("data", data = Images, dtype='float32')
     dset = f.create_dataset("label", data = Contour, dtype='float32')
     dset = f.create_dataset("label2", data = CC, dtype='float32')

