# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:15:46 2016

@author: root
"""
import os
os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 

import scipy.io as sio
f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
"""
for temp in range(Images.shape[0]):
    Images[temp,:,:,:] = Images[temp,:,:,:]-Images[temp,:,:,:].mean()
"""    
Contour = f['Label'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
import numpy as np
CC = f['Contour'].reshape((496,531,1,220)).swapaxes(0,3).swapaxes(1,2).astype('float32')
from numpy.random import choice
from numpy import meshgrid
from scipy import ndimage
def random_sample(L):
    xv, yv = meshgrid(range(L.shape[3]), range(L.shape[2]))
    for stack in range(L.shape[0]):
        l = np.uint(L[stack,0,:,:])
        cc = np.uint(CC[stack,0,:,:])
        cc = ndimage.grey_dilation(cc, size=(5,5))
        a = 512*512
        w = np.zeros((l.max()+np.uint(1)))
        W = np.zeros((l.shape[0],l.shape[1]))
        BW = np.zeros((l.shape[0],l.shape[1])) # CC[stack,0,:,:]
        BW[BW==0] = 10
        for lay in range(l.max()+np.uint(1)):
            w[lay] = (l==lay).sum()
            W[l==lay] = (l==lay).sum()
            if ((l==lay).sum())<a:
                a = (l==lay).sum()
        W = w.sum()/W
        W = W/W.sum()
        x = xv.reshape(BW.shape[0]*BW.shape[1])
        y = yv.reshape(BW.shape[0]*BW.shape[1])
        W = W.reshape(BW.shape[0]*BW.shape[1])
        c = choice(BW.shape[0]*BW.shape[1],a*10,p=W)
        x = x[c]
        y = y[c]
        BW[y,x]=l[y,x]
        BW[cc>0] = l[cc>0]
        L[stack,0,:,:] = BW    
    return L        
#Contour = random_sample(Contour)
n=110
Images = Images[:n,:,:512,:]
Contour = Contour[:n,:,:512,:]
CC = CC[:n,:,:512,:]
for stack in range(CC.shape[0]):
    CC[stack,0,:,:] = ndimage.grey_dilation(np.uint(CC[stack,0,:,:]), size=(2,2))

#Images = Images[:1,:,200:328,200:328]
#Contour = Contour[:1,:,200:328,200:328]

#Images = np.concatenate((Images,Images,Images),axis=1)
#Contour[Contour<1] = 0
#Contour[Contour==1] = 1
#Contour[Contour>1] = 2

Images[Images==255] = 10
Contour = Contour[:,0,:,:]
CC = CC[:,0,:,:]
from skimage.util.shape import view_as_blocks

#Images = np.concatenate((Images[:,:,:256,:],Images[:,:,256:,:]),axis=0)
#Contour = np.concatenate((Contour[:,:256,:],Contour[:,256:,:]),axis=0)
#CC = np.concatenate((CC[:,:256,:],CC[:,256:,:]),axis=0)
Images  = view_as_blocks(Images , block_shape=(1, 1,128,1)).reshape(Images.shape[0]*4,1,496,128).swapaxes(2,3)
Contour= view_as_blocks(Contour , block_shape=(1,128,1)).reshape(Contour.shape[0]*4,496,128).swapaxes(1,2)
CC= view_as_blocks(CC , block_shape=(1,128,1)).reshape(CC.shape[0]*4,496,128).swapaxes(1,2)
#Images = Images[:,:,:,30:374]
#Contour = Contour[:,:,:,30:374]
#CC = CC[:,:,:,30:374]

import h5py
with h5py.File("train.hdf5", "w") as f:
     dset = f.create_dataset("data", data = Images, dtype='float32')
     dset = f.create_dataset("label", data = Contour, dtype='float32')
     dset = f.create_dataset("label2", data = CC, dtype='float32')
#with h5py.File("train_label.hdf5", "w") as f:
#     dset = f.create_dataset("label", data = Contour, dtype='float32')


import matplotlib.pyplot as plt
imgplot = plt.imshow(Images[0,0,:,:])
imgplot = plt.imshow(Contour[0,:,:])
