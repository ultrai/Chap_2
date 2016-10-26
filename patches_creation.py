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
from numpy.random import choice
from sklearn.feature_extraction import image
from scipy.ndimage.morphology import grey_dilation





#tt = np.zeros_like(I)
"""
def patch_select(I,Seg,Contour):
    number_patches_postive = np.int(np.fix(1e6/55/8))
    number_patches_negative = np.int(np.fix(1e6/55/9))
    patch_h = 32 
    patch_w = patch_h
    patches = image.extract_patches_2d(I, (patch_h, patch_w)) # along columns first (531-31)*(496-31)
    patches_GT = image.extract_patches_2d(Seg, (patch_h, patch_w)) # along columns first
    patches_edge = image.extract_patches_2d(Contour, (patch_h, patch_w)) # along columns first
    for lay in range(1,9):
        Contour_temp = (Contour==lay)*1.0
        Seg_temp = Seg==lay
        pos = grey_dilation(Contour_temp, size=[3,3])#*1.0
        neg = Seg_temp
        neg[pos==1]=0
        (y,x) = np.where(pos == 1)
        ind = x-np.int(np.fix(0.5*patch_w))+y*(I.shape[1]-patch_w+1)    
        ind = ind[ind<=((I.shape[0]-patch_h+1)*(I.shape[1]-patch_w+1))]
        C = range(ind.shape[0])
        if len(C)>number_patches_postive:
            C = choice(ind.shape[0],number_patches_postive)
            ind = ind[C]
        (y,x) = np.where(neg == 1)
        ind2 = x-np.int(np.fix(0.5*patch_w))+y*(I.shape[1]-patch_w+1)    
        ind2 = ind2[ind2<=((I.shape[0]-patch_h+1)*(I.shape[1]-patch_w+1))]
        C = range(ind2.shape[0])
        if len(C)>number_patches_negative:
            C = choice(ind2.shape[0],number_patches_negative)
            ind2 = ind2[C]
        if lay == 1:
            Patches = patches[ind,:,:]
            Patches_GT = patches_GT[ind,:,:]
            Patches_edge = patches_edge[ind,:,:]
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
        else:
            Patches = np.concatenate((Patches,patches[ind,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind,:,:]))
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
    Seg_temp = Seg==0
    neg = Seg_temp
    (y,x) = np.where(neg == 1)
    ind2 = x-np.int(np.fix(0.5*patch_w))+y*(I.shape[1]-patch_w+1)    
    ind2 = ind2[ind2<=((I.shape[0]-patch_h+1)*(I.shape[1]-patch_w+1))]
    C = range(ind2.shape[0])
    if len(C)>number_patches_negative:
        C = choice(ind2.shape[0],number_patches_negative)
        ind2 = ind2[C]
    Patches = np.concatenate((Patches,patches[ind2,:,:]))
    Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
    Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
    return(Patches,Patches_GT,Patches_edge)
"""        
def patch_select(I,Seg,Contour):
    number_patches_postive = np.int(np.fix(1e6/55/8))
    number_patches_negative = np.int(np.fix(1e6/55/9))
    patch_h = 32 
    patch_w = patch_h
    patches = image.extract_patches_2d(I, (patch_h, patch_w)) # along columns first (531-31)*(496-31)
    patches_GT = image.extract_patches_2d(Seg, (patch_h, patch_w)) # along columns first
    patches_edge = image.extract_patches_2d(Contour, (patch_h, patch_w)) # along columns first
    Seg= Seg[np.int(np.fix(0.5*patch_h-0.5)):(I.shape[0]-np.int(np.fix(0.5*patch_h-0.5))-1),np.int(np.fix(0.5*patch_h-0.5)):(I.shape[1]-np.int(np.fix(0.5*patch_h-0.5))-1)]
    Contour= Contour[np.int(np.fix(0.5*patch_h-0.5)):(I.shape[0]-np.int(np.fix(0.5*patch_h-0.5))-1),np.int(np.fix(0.5*patch_h-0.5)):(I.shape[1]-np.int(np.fix(0.5*patch_h-0.5))-1)]
    I= I[np.int(np.fix(0.5*patch_h-0.5)):(I.shape[0]-np.int(np.fix(0.5*patch_h-0.5))-1),np.int(np.fix(0.5*patch_h-0.5)):(I.shape[1]-np.int(np.fix(0.5*patch_h-0.5))-1)]
    for lay in range(1,9):
        Contour_temp = (Contour==lay)*1.0
        Seg_temp = Seg==lay
        pos = grey_dilation(Contour_temp, size=[3,3])#*1.0
        neg = Seg_temp
        neg[pos==1]=0
        (y,x) = np.where(pos == 1)
        ind = x+y*(I.shape[1])    
        C = range(ind.shape[0])
        if len(C)>number_patches_postive:
            C = choice(ind.shape[0],number_patches_postive)
            ind = ind[C]
        (y,x) = np.where(neg == 1)
        ind2 = x+y*(I.shape[1])    
        C = range(ind2.shape[0])
        if len(C)>number_patches_negative:
            C = choice(ind2.shape[0],number_patches_negative)
            ind2 = ind2[C]
        if lay == 1:
            Patches = patches[ind,:,:]
            Patches_GT = patches_GT[ind,:,:]
            Patches_edge = patches_edge[ind,:,:]
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
        else:
            Patches = np.concatenate((Patches,patches[ind,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind,:,:]))
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
    Seg_temp = Seg==0
    neg = Seg_temp
    (y,x) = np.where(neg == 1)
    ind2 = x+y*(I.shape[1])    
    C = range(ind2.shape[0])
    if len(C)>number_patches_negative:
        C = choice(ind2.shape[0],number_patches_negative)
        ind2 = ind2[C]
    Patches = np.concatenate((Patches,patches[ind2,:,:]))
    Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
    Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
    return(Patches,Patches_GT,Patches_edge)



f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels = f['Label'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
import numpy as np
Contours = f['Contour'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')

n=55

#Images = Images[:n,:,:,:]
#Labels = Labels[:n,:,:,:]
#Contours = Contours[:n,:,:,:]
for Image_idx in range(n):
    II = Images[Image_idx][0]
    CC = Contours[Image_idx][0]
    SS = Labels[Image_idx][0]
    (I_patch,GT_patch,C_patch) = patch_select(II,SS,CC)
    if Image_idx==0:
        X_train = I_patch
        Y_train = GT_patch
        Y2_train = C_patch
    else:
        X_train = np.concatenate((X_train,I_patch))
        Y_train = np.concatenate((Y_train,GT_patch))
        Y2_train = np.concatenate((Y2_train,C_patch))
        
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


    
               
                   


