from __future__ import division
import os
os.chdir('/home/phani/Data/caffe/examples/FCN')
os.getcwd() 
import sys
sys.path.insert(0,  '/home/phani/Data/caffe/python')

import caffe
import numpy as np
import time
caffe.set_mode_gpu()#caffe.set_mode_gpu()
#caffe.set_device(0)

solver = caffe.get_solver('FCN_solver.prototxt')
# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
t = time.time()
solver.solve()
elapsed = time.time() - t
print elapsed