import os
os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 

#caffe_root = '../caffe-crfrnn/'
import sys
#sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0,  '../../python')

import numpy as np

import caffe
net = caffe.Net('FCN2_deploy.prototxt', 'FCN2_iter_50000.caffemodel', caffe.TEST)
#caffe.set_mode_gpu()
#caffe.set_device(1)

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

from skimage.graph import route_through_array
def short_path(Prob):
    L = np.zeros((Prob.shape[0],Prob.shape[1]))
    Idx = np.zeros((Prob.shape[2]-1,Prob.shape[1]))
    #padd = np.ones((Prob.shape[0],1))
    for lay in range(1,Prob.shape[2]):
        I = Prob[:,:,lay-1]
        II = np.ones((I.shape[0],I.shape[1]+2))
        II[:,1:-1]=I
        #II = np.array(np.concatenate((padd,I,padd),axis=1)) #+0.00000001
        indices, weight = route_through_array(1-II, (0, 0), (II.shape[0]-1,II.shape[1]-1), geometric=False)
        indices = np.array(indices).T
        path = np.zeros_like(II)
        path[indices[0], indices[1]] = 1.0
        L = L+path[:,1:-1]*(lay)
        #ii = indices[1,:]
        #ii = ii[ii>0]
        #ii = ii[ii<(path.shape[1]-1)]
        Idx[lay-1,:] = np.argmax(path[:,1:-1],axis=0)   
    return(Idx)
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

#im = Images[0,:,200:328,200:328]
from skimage.util.shape import view_as_blocks

Images  = view_as_blocks(Images , block_shape=(1, 1,128,1)).reshape(Images.shape[0]*4,1,496,128).swapaxes(2,3)
Labels = view_as_blocks(Labels[:,0,:,:] , block_shape=(1,128,1)).reshape(Labels.shape[0]*4,496,128).swapaxes(1,2)
Labels_AD = view_as_blocks(Labels_AD[:,0,:,:] , block_shape=(1,128,1)).reshape(Labels_AD.shape[0]*4,496,128).swapaxes(1,2)

Labels_hat = np.zeros((Labels.shape))

for Idx in range(Images.shape[0]):
    #Idx=0
    im = Images[Idx,:,:,:]
    im[im==255]   = 10
    in_ = im.reshape(1, *im.shape)#.swapaxes(1,2)
    net.blobs['data'].data[...] = in_
    out = net.forward()['prob1'][0,:,:,:].swapaxes(0,2)
    Labels_hat[Idx,:,:] = out.argmax(axis=2).swapaxes(0,1)

Predictions = np.zeros((8,Images.shape[2],Images.shape[0]))
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
for Idx in range(Images.shape[0]):
    im = Images[Idx,:,:,:]
    im[im==255]   = 10
    in_ = im.reshape(1, *im.shape)#.swapaxes(1,2)
    net.blobs['data'].data[...] = in_
    ou = net.forward()['prob1'][0,:,:,:].swapaxes(0,2)
    out = np.zeros_like(ou)
    ou = ou.argmax(axis=2)
    for l in range(out.shape[2]):
        #out[:,:,l] = feature.canny(1.0*(ou>l), sigma=3)
        out[:,:,l] = sobel(1.0*(ou>l))
    Predictions[:,:,Idx] = short_path(out)
Labels_hatC = maps2labels(Images[:,0,:,:].swapaxes(0,2),Predictions)[0].swapaxes(0,2)-1
CM = np.zeros((Images.shape[0],9))
CMc = np.zeros((Images.shape[0],9))
CMd = np.zeros((Images.shape[0],9))

from sklearn.metrics import confusion_matrix

for Idx in range(Images.shape[0]):
    L = (Labels[Idx,:,:]).reshape(Images.shape[2]*Images.shape[3],)
    L_hat = (Labels_hat[Idx,:,:]).reshape(Images.shape[2]*Images.shape[3],)
    L_hatC = (Labels_hatC[Idx,:,:]).reshape(Images.shape[2]*Images.shape[3],)
    L_hatD = (Labels_AD[Idx,:,:]).reshape(Images.shape[2]*Images.shape[3],)
    cm = confusion_matrix( L, L_hat)
    CM[Idx,:] = np.diag(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    cm = confusion_matrix( L, L_hatC)
    CMc[Idx,:] = np.diag(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    cm = confusion_matrix( L, L_hatD)
    CMd[Idx,:] = np.diag(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    
CM

Labels_hatC = Labels_hatC.reshape(Labels_hatC.shape[0],1,Labels_hatC.shape[1],Labels_hatC.shape[2])[:,0,:,:]
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#cm = confusion_matrix( Contour[0,0,:,:].reshape(512*496,),  Contour_hat[0,0,:,:].reshape(512*496,))
imgplot = plt.imshow(Labels[0,:,:])
imgplot = plt.imshow(Labels_hatC[0,:,:])

Labelss = Labels.reshape(Images.shape[0]*Images.shape[2]*Images.shape[3],)
#Labels_hat = Labels
Labels_h =  Labels_hat.reshape(Images.shape[0]*Images.shape[2]*Images.shape[3],)
cm = confusion_matrix( Labelss, Labels_h)
Cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
Labels_h =  Labels_hatC.reshape(Images.shape[0]*Images.shape[2]*Images.shape[3],)
cm = confusion_matrix( Labelss, Labels_h)
Cm_normalizedC = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
Labels_AD =  Labels_AD.reshape(Images.shape[0]*Images.shape[2]*Images.shape[3],)
cm = confusion_matrix( Labelss, Labels_AD)
Cm_normalized_AD = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

import matplotlib.pyplot as plt
plt.imshow(Cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.imshow(Cm_normalizedC<Cm_normalized_AD, interpolation='nearest', cmap=plt.cm.Blues)

#imgplot = plt.imshow(out.argmax(axis=2))
#imgplot = plt.imshow(out[:,:,1])
