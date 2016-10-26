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
os.chdir('C:/Users/Ultrai`s solitude/Downloads/2015_BOE_Chiu')

import scipy.io as sio
from numpy.random import choice
from sklearn.feature_extraction import image
from scipy.ndimage.morphology import grey_dilation





#tt = np.zeros_like(I)
def patch_select(I,Seg,Contour):
    number_patches_postive = 1e6/55/8
    number_patches_negative = 1e6/55/9
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
        c = range(x.shape[0])
        if len(c)>number_patches_postive:
            c = choice(x.shape[0],number_patches_postive)
        ind = x[c]-np.fix(0.5*patch_w)+y[c]*(I.shape[1]-patch_w+1)    
        (y,x) = np.where(neg == 1)
        c = range(x.shape[0])
        if len(c)>number_patches_negative:
            c = choice(x.shape[0],number_patches_negative)
        ind2 = x[c]-np.fix(0.5*patch_w)+y[c]*(I.shape[1]-patch_w+1)    
        if lay == 1:
            Patches = patches[ind,:,:]
            Patches_GT = patches_GT[ind,:,:]
            Patches_edge = patches_edge[ind,:,:]
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
        else:
            Patches = np.concatenate(Patches,patches[ind,:,:])
            Patches_GT = np.concatenate(Patches_GT,patches_GT[ind,:,:])
            Patches_edge = np.concatenate(Patches_edge,patches_edge[ind,:,:])
            Patches = np.concatenate((Patches,patches[ind2,:,:]))
            Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
            Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
        Seg_temp = Seg==0
        neg = Seg_temp
        (y,x) = np.where(neg == 1)
        c = range(x.shape[0])
        if len(c)>number_patches_negative:
            c = choice(x.shape[0],number_patches_negative)
        ind2 = x[c]-np.fix(0.5*patch_w)+y[c]*(I.shape[1]-patch_w+1)    
        Patches = np.concatenate((Patches,patches[ind2,:,:]))
        Patches_GT = np.concatenate((Patches_GT,patches_GT[ind2,:,:]))
        Patches_edge = np.concatenate((Patches_edge,patches_edge[ind2,:,:])) 
        return(Patches,Patches_GT,Patches_edge)
        

f = sio.loadmat('Data.mat')
Images = f['Images'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')
Labels = f['Label'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')-1
import numpy as np
Contours = f['Contour'].reshape((496,531,1,110)).swapaxes(0,3).swapaxes(1,2).astype('float32')


II = Images[0][0]
CC = Contours[0][0]
SS = Labels[0][0]

(I_patch,GT_patch,C_patch) = patch_select(II,SS,CC)

n=55

Images = Images[:n,:,:512,:]
Labels = Labels[:n,:,:512,:]
Contours = Contours[:n,:,:512,:]

    
               
                   


