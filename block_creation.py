# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:45:42 2016

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:07:00 2016

@author: Ultrai`s solitude
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.chdir('/home/mict/Desktop/Chap_2/')

import scipy.io as sio
import numpy as np
from numpy.random import choice
from sklearn.feature_extraction import image
from scipy.ndimage.morphology import grey_dilation
from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_blocks


f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels = f['Label'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Contours = f['Contour'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
"""
I = Images[0][0]
Seg = Labels[0][0]
Contour = Contours[0][0]
II_temp = np.lib.pad(np.ones((2,2)), ((3, 2), (0, 0)),  'constant', constant_values=(0))
"""         
def block_select(I,Seg,Contour):
    patch_h = np.float(64)
    patch_w = patch_h
    delta = 8
    step = (1.0/delta)*patch_h
    sz = np.float64(np.array(I.shape))
    pad_w = np.ceil(sz[1]/patch_w)*patch_w-sz[1]
    pad_h = np.ceil(sz[0]/patch_h)*patch_h-sz[0]

    I_temp = np.lib.pad(I, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
    Seg_temp = np.lib.pad(Seg, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
    Contour_temp = np.lib.pad(Contour, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
    
    Patches_GT = view_as_blocks(Seg_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
    Patches_edge = view_as_blocks(Contour_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
    Patches = view_as_blocks(I_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
    
    
    if delta>1:
        for Delta in range(1,delta):
            I = I[step:,:]
            Seg = Seg[step:,:]
            Contour = Contour[step:,:]
            sz = np.float64(np.array(I.shape))
            pad_w = np.ceil(sz[1]/patch_w)*patch_w-sz[1]
            pad_h = np.ceil(sz[0]/patch_h)*patch_h-sz[0]
            I_temp = np.lib.pad(I, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
            Seg_temp = np.lib.pad(Seg, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
            Contour_temp = np.lib.pad(Contour, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
            Seg_temp = view_as_blocks(Seg_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
            Contour_temp = view_as_blocks(Contour_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
            I_temp = view_as_blocks(I_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
            Patches = np.concatenate((Patches,I_temp))
            Patches_GT = np.concatenate((Patches_GT,Seg_temp))
            Patches_edge = np.concatenate((Patches_edge,Contour_temp)) 
            for Delta2 in range(1,delta):
                I = I[step:,:]
                Seg = Seg[step:,:]
                Contour = Contour[step:,:]
                sz = np.float64(np.array(I.shape))
                pad_w = np.ceil(sz[1]/patch_w)*patch_w-sz[1]
                pad_h = np.ceil(sz[0]/patch_h)*patch_h-sz[0]
                I_temp = np.lib.pad(I, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
                Seg_temp = np.lib.pad(Seg, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
                Contour_temp = np.lib.pad(Contour, ((np.int(np.fix(pad_h*0.5)), np.int(pad_h-np.fix(pad_h*0.5))), (np.int(np.fix(pad_w*0.5)), np.int(pad_w-np.fix(pad_w*0.5)))), 'reflect') 
                Seg_temp = view_as_blocks(Seg_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
                Contour_temp = view_as_blocks(Contour_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
                I_temp = view_as_blocks(I_temp, block_shape=(patch_h,patch_w)).reshape(np.int(1.0*I_temp.shape[0]*I_temp.shape[1]/patch_w/patch_h),np.int(patch_h),np.int(patch_w))
                Patches = np.concatenate((Patches,I_temp))
                Patches_GT = np.concatenate((Patches_GT,Seg_temp))
                Patches_edge = np.concatenate((Patches_edge,Contour_temp)) 
    return(Patches,Patches_GT,Patches_edge)



f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels = f['Label'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
Contours = f['Contour'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')

n=55

#Images = Images[:n,:,:,:]
#Labels = Labels[:n,:,:,:]
#Contours = Contours[:n,:,:,:]
for Image_idx in range(n):
    II = Images[Image_idx][0]
    CC = Contours[Image_idx][0]
    SS = Labels[Image_idx][0]
    (I_patch,GT_patch,C_patch) = block_select(II,SS,CC)
    if Image_idx==0:
        X_train = I_patch
        Y_train = GT_patch
        Y2_train = C_patch
    else:
        X_train = np.concatenate((X_train,I_patch))
        Y_train = np.concatenate((Y_train,GT_patch))
        Y2_train = np.concatenate((Y2_train,C_patch))
Y_train = Y_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])        
Y2_train = Y2_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])        
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])        
"""
for Image_idx in range(n,110):
    II = Images[Image_idx][0]
    CC = Contours[Image_idx][0]
    SS = Labels[Image_idx][0]
    (I_patch,GT_patch,C_patch) = patch_select(II,SS,CC)
    if Image_idx==n:
        X_test = I_patch
        Y_test = GT_patch
        Y2_test = C_patch
    else:
        X_test = np.concatenate((X_test,I_patch))
        Y_test = np.concatenate((Y_test,GT_patch))
        Y2_test = np.concatenate((Y2_test,C_patch))
"""
X_test = Images[n:,:,:,:]
Y_test = Labels[n:,:,:,:]
Y2_test = Contours[n:,:,:,:]

Images_test = Images[n:,:,9:521,24:472]
Labels_test = Labels[n:,0,9:521,24:472]
Contours_test = Contours[n:,0,9:521,24:472]

block = 64
Images_test  = view_as_blocks(Images_test , block_shape=(1, 1,block,block))
X_test = Images_test.reshape(Images_test.shape[0]*Images_test.shape[2]*Images_test.shape[3],1,Images_test.shape[6],Images_test.shape[7]).swapaxes(2,3)
Y_test = view_as_blocks(Labels_test , block_shape=(1,block,block)).reshape(X_test.shape[0],X_test.shape[2],X_test.shape[3]).swapaxes(1,2)
Y2_test = view_as_blocks(Contours_test , block_shape=(1,block,block)).reshape(X_test.shape[0],X_test.shape[2],X_test.shape[3]).swapaxes(1,2)
for stack in range(Y2_train.shape[0]):
    Y2_train[stack,0,:,:]  = grey_dilation(np.uint(Y2_train[stack,0,:,:]), size=(2,2))
    

import h5py
with h5py.File("train.hdf5", "w") as f:
     dset = f.create_dataset("data", data = X_train, dtype='float32')
     dset = f.create_dataset("label", data = Y_train, dtype='float32')
     dset = f.create_dataset("label2", data = Y2_train, dtype='float32')
with h5py.File("test.hdf5", "w") as f:
     dset = f.create_dataset("data", data = X_test, dtype='float32')
     dset = f.create_dataset("label", data = Y_test, dtype='float32')
     dset = f.create_dataset("label2", data = Y2_test, dtype='float32')

    
               
                   


