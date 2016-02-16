import caffe as cf
import os
import sys
from utils_birds import *
sys.path.insert(0, '/home/jsotaloram/software/caffe/python')
import numpy as np
from pylab import *
#%matplotlib inline


#https://github.com/BVLC/caffe/blob/master/examples/03-fine-tuning.ipynb
dataset_path= '/data1/birds/'
birdsFolder = "/data1/birds"
birdsLabels = os.listdir(birdsFolder)
dict_cats = {0:'wood_duck/wod', 1:'egret/egr',2:'owl/owl', 3:'toucan/tou',4:'mandarin/man',5:'puffin/puf'}
#dict_cats = {(k,v) for (k,v) in [(birdsLabels[x],x) for x in range(len(birdsLabels))]}

cf.set_device(0)
cf.set_mode_gpu()

modelAlexNet = "bvlc_alexnet.caffemodel" #assuming is in the same folder
deployProtoTxt = "deploy.prototxt" 
imageNetLabels = np.loadtxt("synset_words.txt", str, delimiter='\t')
splitSamples = np.loadtxt("splitSamples.txt", str, delimiter=';')

niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)
# We create a solver that fine-tunes from a previously trained network.
solver = cf.SGDSolver('/home/jsotaloram/software/caffe/models/finetune_flickr_style/solver.prototxt')
solver.net.copy_from('bvlc_alexnet.caffemodel')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    #scratch_solver.step(1)
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    #scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it])
print 'done finnetunning!'

#Now lets compute the performance measures





