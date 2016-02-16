
# coding: utf-8

# In[29]:

import caffe as cf
import cPickle as pickle
from utils_birds import loadSamples
import numpy as np
from caffe import layers as cfL
from caffe import params as cfP
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



# In[31]:

#reading the extracted feats of alexnet
with(open('/home/jsotaloram/active_learning/exp_birds/feats_alexnet_birdsDS.pkl','rb')) as f:
    birds_data = pickle.load(f)
#birds_data contains alexNet feats fc6 and fc7 for all 600 samples: 
#Let's compute the performance measures, first partitionate as required
samples = loadSamples()
splitSamples = np.loadtxt("splitSamples.txt", str, delimiter=';')
birdsFolder = "/data1/birds"

birdsLabels = os.listdir(birdsFolder)

woodSamples = np.array(splitSamples[1:, 0], dtype=int) -1
egretSamples = 100*1 + np.array(splitSamples[1:, 1], dtype=int) - 1
owlSamples = 100*2 + np.array(splitSamples[1:, 2], dtype=int) - 1
toucanSamples = 100*3 + np.array(splitSamples[1:, 3], dtype=int) - 1
puffinSamples = 100*4 + np.array(splitSamples[1:, 4], dtype=int) - 1
mandarinSamples = 100*5 + np.array(splitSamples[1:, 5], dtype=int) - 1
trainIdx = np.concatenate((
        woodSamples[0:20],
        egretSamples[0:20],
        owlSamples[0:20],
        toucanSamples[0:20],
        puffinSamples[0:20],
        mandarinSamples[0:20],
        ))

crossIdx = np.concatenate((
        woodSamples[20:50],
        egretSamples[20:50],
        owlSamples[20:50],
        toucanSamples[20:50],
        puffinSamples[20:50],
        mandarinSamples[20:50],
        ))

testIdx = np.concatenate((
        woodSamples[50:100],
        egretSamples[50:100],
        owlSamples[50:100],
        toucanSamples[50:100],
        puffinSamples[50:100],
        mandarinSamples[50:100],
        ))


# In[32]:

samples
dict_cats_nameOnly = {0:'wood_duck', 1:'egret',2:'owl', 3:'toucan',4:'mandarin',5:'puffin'}

inv_bird_map_nameOnly = {v: k for k, v in dict_cats_nameOnly.items()}

all_samples_cats = []
for sample in samples:
    all_samples_cats.append(inv_bird_map_nameOnly[sample[0]])
len(all_samples_cats)

yTrainLfc6_labels = [all_samples_cats[i] for i in trainIdx]
yCrossLfc6_labels = [all_samples_cats[i] for i in crossIdx]
yTestLfc6_labels =  [all_samples_cats[i] for i in testIdx]
len(yTrainLfc6_labels)


# In[33]:

XTrainLfc6 = birds_data[0][trainIdx]
yTrainLfc6 = samples[trainIdx, 0]
XCrossLfc6 = birds_data[0][crossIdx]
yCrossLfc6 = samples[crossIdx, 0]
XTestLfc6 = birds_data[0][testIdx]
yTestLfc6 = samples[testIdx, 0]
xTrainCaffe = [XTrainLfc6 ]
len(yTestLfc6)
#Asserting that we are mapping good the name to the category
yTestLfc6[0] == dict_cats_nameOnly[yTestLfc6_labels[0]]
#yTestLfc6_labels


# In[34]:

yCrossLfc6.shape


# In[35]:

batch_size = 20
solverPrototxt='/home/jsotaloram/active_learning/exp_birds/solver_s_birds.prototxt'
deployPrototxt='/home/jsotaloram/active_learning/exp_birds/train_birds_small.prototxt'
cf.set_mode_gpu()
active_birds_net = cf.Net(deployPrototxt, cf.TRAIN)


# In[36]:

all_shapes = [(k, v.data.shape) for k, v in active_birds_net.blobs.items()]
num_hidden_neurons = all_shapes[2][1][1] #the first dimension of shape of fc7 layer
num_hidden_neurons


# In[37]:

train_size = 120
val_size = 180
test_size = 300
data4D = np.zeros([train_size ,1,1,4096],np.float32)
data4DL = np.zeros([train_size ,1,1,1], np.float32)
data4D_val = np.zeros([val_size ,1,1,4096],np.float32)
data4DL_val = np.zeros([val_size ,1,1,1], np.float32)
data4D_test = np.zeros([test_size ,1,1,4096],np.float32)
data4DL_test = np.zeros([test_size ,1,1,1], np.float32)

full_idx_range = np.arange(train_size)
full_idx_range_val = np.arange(val_size)
full_idx_range_test = np.arange(test_size)

#print(full_idx_range)
np.random.shuffle(full_idx_range)
np.random.shuffle(full_idx_range_val)
np.random.shuffle(full_idx_range_test)

#print(full_idx_range)

Xrs = XTrainLfc6.reshape(train_size,1,1,4096)
Yrs = np.array(yTrainLfc6_labels).reshape(train_size,1,1,1)

Xval = XCrossLfc6.reshape(val_size,1,1,4096)
Yval = np.array(yCrossLfc6_labels).reshape(val_size,1,1,1)

Xtest = XTestLfc6.reshape(test_size,1,1,4096)
Ytest = np.array(yTestLfc6_labels).reshape(test_size,1,1,1)

data4D[0:120,:,:,:]  = [Xrs[train_idx] for train_idx in full_idx_range]
data4DL[0:120,:,:,:] = [Yrs[train_idx] for train_idx in full_idx_range]

data4D_val[0:180,:,:,:]  = [Xval[val_idx] for val_idx in full_idx_range_val]
data4DL_val[0:180,:,:,:] = [Yval[val_idx] for val_idx in full_idx_range_val]

data4D_test[0:300,:,:,:]  = [Xtest[test_idx] for test_idx in full_idx_range_test]
data4DL_test[0:300,:,:,:] = [Ytest[test_idx] for test_idx in full_idx_range_test]

print Xrs.shape
print data4DL.shape


#active_birds_net.blobs['data'].data[...] = data4D[0:10] #hardcoding the first batch
#active_birds_net.blobs['label'].data[...] = data4DL[0:10]#yTrainLfc6_labels[0:10]


# In[ ]:

lines_solver = []
f = open(solverPrototxt,'r')
for line in f:
    lines_solver.append(line)

learning_rates_pool = [0.0001,0.001,0.01,0.1,1,10,100]
decay_mult = [0.0001,0.001,0.01,0.1,1,10,100]


save_protos_path = '/home/jsotaloram/active_learning/exp_birds/protos/'
hyper_params_combs = [] 
for lr_item in learning_rates_pool:
    for dec_item in decay_mult:
        #print 'combination of lr: ' + str(lr_item) +' and decay: ' +str(dec_item) + '\n'
        hyper_params_combs.append((lr_item,dec_item))

for (lr_i,wd_i) in hyper_params_combs:
    temp_proto = open(save_protos_path+'solver_proto_lr'+str(lr_i)+'_wd'+str(wd_i)+'_birds.prototxt','w')
    temp_proto.write('net: ' + '\"'+ save_protos_path+'proto_lr'+ str(lr_i)+'_wd' + str(wd_i)+'_birds.prototxt\"' + '\n')
    temp_proto.writelines([line_p for line_p in lines_solver[1:len(lines_solver)]])

#[line_p for line_p in lines_solver[0:10]]


# In[ ]:

niter = 1000
for (lr_i,wd_i) in [(100,100)]:#hyper_params_combs:
    temp_solver_path = save_protos_path+'solver_proto_lr'+str(lr_i)+'_wd'+str(wd_i)+'_birds.prototxt'
    print 'computing performance for solver: ' + temp_solver_path
    solver = cf.SGDSolver(temp_solver_path)
    train_loss = np.zeros(niter)
    scratch_train_loss = np.zeros(niter)
    num_hidden = num_hidden_neurons

    solver.net.set_input_arrays(data4D, data4DL)
    solver.test_nets[0].set_input_arrays(data4D_test,data4DL_test)

#It seems that labels should be of dimension (n, 1, 1, 1) for memory layer as well.


    for it in range(niter):
        solver.step(1)  # SGD by Caffe                                                                                                                     
        #scratch_solver.step(1)                                                                                                                            
        # store the train loss                                                                                                                             
        train_loss[it] = solver.net.blobs['loss'].data
        #scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data                                                                                    
        #if it % 10 == 0:
            #print 'iter %d, Training_Birds: hidden_%d_MLP_loss=%f' % (it,num_hidden_neurons, train_loss[it])
    #print 'done Training!'


    #reporting classification meassures of the small net:
    test_net = solver.test_nets[0] # more than one test net is supported

    test_net.set_input_arrays(data4D_test, data4DL_test)

    pred_labels_test_birds = []

    output = test_net.forward()

    cur_lbls = []
    for i in range(300):
        pred_labels_test_birds.append(test_net.blobs['prob'].data[i].flatten().argsort()[-1:-2:-1][0])



#CMSvmLfc7 = confusion_matrix(yTestLfc6_labels, pred_labels_test_birds, labels=birdsLabels)
#plt.figure(figsize=(8, 8))
#plot_confusion_matrix(CMSvmLfc7, labels=birdsLabels)

    #print(classification_report(yTestLfc6_labels, pred_labels_test_birds))
    cr_f1 = classification_report(yTestLfc6_labels, pred_labels_test_birds).split()
    print 'f1 so far: ' + str(cr_f1[-2]) + 'lr_i: ' + str(lr_i) + 'wd_i: ' + str(wd_i)


#print solver.net.blobs['label'].data[29]
#output = solver.net.forward()
#print data4DL_test[29]
#output['prob'][29]

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=birdsLabels):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
