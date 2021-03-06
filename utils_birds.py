import caffe as cf
import os
import sys
import numpy as np
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import caffe as cf
import os
from scipy.special import expit as sigmoid
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import scipy.optimize


#%matplotlib inline
birdsFolder = "/data1/birds"
birdsLabels = os.listdir(birdsFolder)


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


def loadSamples():
    labels = os.listdir(birdsFolder)
    dataset = np.empty((0,2))
    
    for label in labels:
        samples = sorted(os.listdir(os.path.join(birdsFolder, label)))
        for sample in samples:
            path = os.path.join(birdsFolder, label, sample)
            if os.path.isfile(path):
                dataset = np.concatenate((dataset, np.array([[label, path]])), axis=0)                
    return dataset



def predict(net, samples):
    pred = np.empty((0,1), dtype=int)
    outputLfc6 = np.empty((0,4096))
    outputLfc7 = np.empty((0,4096))
    transformer = cf.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/opt/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    for sample in samples:
        net.blobs['data'].data[...] = transformer.preprocess('data', cf.io.load_image(sample[1]))
        out = net.forward()
        pred = np.concatenate((pred, [net.blobs['prob'].data[0].flatten().argsort()[-1:-2:-1]]))
        outputLfc6 = np.concatenate((outputLfc6, [net.blobs['fc6'].data[0]]))
        outputLfc7 = np.concatenate((outputLfc7, [net.blobs['fc7'].data[0]]))
    
    return pred, outputLfc6, outputLfc7

def create_train_val_test_files(name_train, name_val, name_test):
    f = open(name_train,'w')
    g = open(name_val,'w')
    h = open(name_test,'w')
    for i in range (1,21): #training samples according to http://www-cvr.ai.uiuc.edu/ponce_grp/data/birds/birds_f_numbers.txt
        row_ids = [item for item in splitSamples[i]]
        for j in range(len(row_ids)):
            f.write(dataset_path + dict_cats[j] + row_ids[j].zfill(3) + '.jpg '+ str(j) +'\n')
    for i in range (21,50): #val samples according to http://www-cvr.ai.uiuc.edu/ponce_grp/data/birds/birds_f_numbers.txt
        row_ids = [item for item in splitSamples[i]]
        for j in range(len(row_ids)):
            g.write(dataset_path + dict_cats[j] + row_ids[j].zfill(3) + '.jpg '+ str(j) +'\n')
    for i in range (50,100): #test samples according to http://www-cvr.ai.uiuc.edu/ponce_grp/data/birds/birds_f_numbers.txt
        row_ids = [item for item in splitSamples[i]]
        for j in range(len(row_ids)):
            h.write(dataset_path + dict_cats[j] + row_ids[j].zfill(3) + '.jpg '+ str(j) +'\n')
    f.close()
    g.close()
    h.close()


def generate_protos(gendir_path, hyper_params):
    solverPrototxt='/home/jsotaloram/active_learning/exp_birds/solver_s_birds.prototxt'
    deployPrototxt='/home/jsotaloram/active_learning/exp_birds/train_birds_small.prototxt'

    lines_solver = []
    save_protos_path = '/home/jsotaloram/active_learning/exp_birds/protos/'
    lines_proto = []
    f = open(deployPrototxt,'r')
    for line in f:
        lines_proto.append(line)
    
    [line_p for line_p in lines_proto[0:10]]
    f = open(solverPrototxt,'r')
    for line in f:
        lines_solver.append(line)

    for (lr_i,wd_i) in hyper_params:
        print 'generating prototxt for LR: ' + str(lr_i) + ' and WD: ' + str(wd_i)
        temp_proto = open(gendir_path+'proto_lr'+str(lr_i)+'_wd'+str(wd_i)+'_birds.prototxt','w')
        temp_proto.writelines([line_p for line_p in lines_proto[0:40]])
        temp_proto.write('    lr_mult: '+str(lr_i)+'\n')
        temp_proto.write('    decay_mult: '+str(wd_i)+'\n')
        temp_proto.writelines([line_p for line_p in lines_proto[42:44]])
        temp_proto.write('    lr_mult: '+str(lr_i)+'\n')
        temp_proto.write('    decay_mult: '+str(wd_i)+'\n')
        temp_proto.writelines([line_p for line_p in lines_proto[46:69]])
        temp_proto.write('    lr_mult: '+str(lr_i)+'\n')
        temp_proto.write('    decay_mult: '+str(wd_i)+'\n')
        temp_proto.writelines([line_p for line_p in lines_proto[71:73]])
        temp_proto.write('    lr_mult: '+str(lr_i)+'\n')
        temp_proto.write('    decay_mult: '+str(wd_i)+'\n')
        temp_proto.writelines([line_p for line_p in lines_proto[75:len(lines_proto)]])


def generate_hyperparams_comb():
    learning_rates_pool = [0.0001,0.0003, 0.001,0.005,0.01,0.1,0.5,0.8,1,2,4]
    decay_mult_pool = [0.0001,0.001,0.005, 0.01,0.1,0.5, 1,5,2,10,100]
    hyper_params_combs = []
    for lr_item in learning_rates_pool:
        for dec_item in decay_mult_pool:
        #print 'combination of lr: ' + str(lr_item) +' and decay: ' +str(dec_item) + '\n'
            hyper_params_combs.append((lr_item,dec_item))
    return hyper_params_combs
