import os
os.chdir('/home/phani/Data/caffe/examples/FCN')

os.getcwd() 
import glob
ls = sorted(glob.glob('data/*.mat'))
import scipy.io as sio
import numpy as np
nan = np.nan
isnan = np.isnan

def interp_nan(layers):
    s=layers.shape
    L = np.zeros([s[0],s[1],s[2]])
#    L = zeros(size(layers));
    for temp in range(s[2]):
        #for temp = 1:size(layers,3)
        l = layers[:,:,temp]
        #l = layers(:,:,temp);
        for layer in range(s[0]):
        #for layer = 1:size(l,1)
            A = l[layer,:]
            ok = ~np.isnan(A)
            xp = ok.ravel().nonzero()[0]
            fp = A[~np.isnan(A)]
            x  = np.isnan(A).ravel().nonzero()[0]
            A[np.isnan(A)] = np.interp(x, xp, fp)
            l[layer,:] = A
        L[:,:,temp] = l
    return L

def maps2labels(I,L):
    list = []
    Label = np.zeros((I.shape[0],I.shape[1],I.shape[2]), dtype='uint8')
    Label2 = np.zeros((I.shape[0],I.shape[1],I.shape[2]), dtype='uint8')
    for scan in range(I.shape[2]):
        image = I[:,:,scan]
        s = image.shape
        l = L[:,:,scan]
        if np.sum(~np.isnan(l.sum(0)))>50:
            list.append(scan)
            l = np.round(l)
            l[l<3]=3
            l[l>s[0]] = s[0]
            layers = np.zeros((l.shape[0]+1,s[1],2), dtype='int')
            layers[1::,:,0] = l
            layers[:-1,:,1] = l
            layers[-1,:,1] = s[0]
            label = np.zeros((s[0],s[1]))
            label2 = np.zeros((s[0],s[1]))
            for col in range(s[1]):
                for lay in range(layers.shape[0]):
                    label[layers[lay,col,0]:layers[lay,col,1],col]=lay+1
                    label2[layers[lay,col,0],col]=lay
            Label[:,:,scan] = label
            Label2[:,:,scan] = label2
    return (Label,Label2,I,list)    

for x in range(len(ls)):
    #x=0
    data = sio.loadmat(ls[x])
    I = data['images']    #data.keys()
    L1 = data['manualLayers1']
    L2 = data['manualLayers2']
    AN = data['automaticLayersNormal']
    AD = data['automaticLayersDME']
    nan_check = np.sum(np.sum(~isnan(L1),axis=0),axis=0)
    I = I[:,:,nan_check>0]
    Label1 = L1[:,:,nan_check>0]
    Label2 = L2[:,:,nan_check>0]
    AN = AN[:,:,nan_check>0]
    AD = AD[:,:,nan_check>0]
    L1 = interp_nan(Label1)
    L2 = interp_nan(Label2)
    AN = interp_nan(AN)
    AD = interp_nan(AD)
    L_M1,C_M1,Im,temp = maps2labels(I,L1-1) # as python is 0 index
    L_M2,C_M2,temp,temp = maps2labels(I,L2-1) # as python is 0 index
    L_AN,C_AN,temp,temp = maps2labels(I,AN-1) # as python is 0 index
    L_AD,C_AD,temp,temp = maps2labels(I,AD-1) # as python is 0 index
    L,C,temp,temp = maps2labels(I,0.5*(L1+L2)-1) # as python is 0 index
    if x==0:
        Labels = np.concatenate((L_M1,L_M2),axis=2)
        Contours = np.concatenate((C_M1,C_M2),axis=2)
        Images = np.concatenate((Im,Im),axis=2)
        Labels_AN = np.concatenate((L_AN,L_AN),axis=2)
        Contours_AN = np.concatenate((C_AN,C_AN),axis=2)
        Labels_AD = np.concatenate((L_AD,L_AD),axis=2)
        Contours_AD = np.concatenate((C_AD,C_AD),axis=2)
    else:
        Labels = np.concatenate((Labels,L_M1,L_M2),axis=2)
        Contours = np.concatenate((Contours,C_M1,C_M2),axis=2)
        Images = np.concatenate((Images,Im,Im),axis=2)
        Labels_AN = np.concatenate((Labels_AN,L_AN,L_AN),axis=2)
        Labels_AD = np.concatenate((Labels_AD,L_AD,L_AD),axis=2)
        Contours_AN = np.concatenate((Contours_AN,C_AN,C_AN),axis=2)
        Contours_AD = np.concatenate((Contours_AD,C_AD,C_AD),axis=2)
Labels = Labels[:,120:651,:]
Contours = Contours[:,120:651,:]
Images = Images[:,120:651,:]
Labels_AN = Labels_AN[:,120:651,:]
Contours_AN = Contours_AN[:,120:651,:]
Labels_AD = Labels_AD[:,120:651,:]
Contours_AD = Contours_AD[:,120:651,:]

import scipy.io as sio
sio.savemat('Data.mat', {'Images': Images, 'Label': Labels, 'Contour': Contours, 'Label_AN': Labels_AN, 'Contour_AN': Contours_AN, 'Label_AD': Labels_AD, 'Contour_AD': Contours_AD})
import matplotlib.pyplot as plt
imgplot = plt.imshow(Contours[:,:,0])    

