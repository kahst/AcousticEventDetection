#!/usr/bin/env python

print "HANDLING IMPORTS..."

import warnings
warnings.filterwarnings('ignore')

import os
import time
import operator

import traceback
import numpy as np
import pickle

import theano

from lasagne import random as lasagne_random
from lasagne import layers as l

import scipy.io.wavfile as wave

import AED_spec as spectrogram
import utils.batch_generator as bg

print "...DONE!"

######################## CONFIG #########################
#Fixed random seed
RANDOM_SEED = 1337
RANDOM = np.random.RandomState(RANDOM_SEED)
lasagne_random.set_rng(RANDOM)

#Dataset params
TEST_DIR = 'dataset/test/'

#Pre-trained model params
MODEL_PATH = 'model/'
TRAINED_MODEL = 'AED_Example_Run_model.pkl'

#Testing params
BATCH_SIZE = 32
SPEC_LENGTH = 3
SPEC_OVERLAP = 2
CONFIDENCE_THRESHOLD = 0.0001
MAX_PREDICTIONS = 10

################### AUDIO PROCESSING ####################
def parseTestSet():    

    #get list of test files
    test = []
    test_classes = [os.path.join(TEST_DIR, tc) for tc in sorted(os.listdir(TEST_DIR))]
    for tc in test_classes:
        test += [os.path.join(tc, fpath) for fpath in os.listdir(tc)]
    test = test

    #get class label for every test sample
    gt = {}
    for path in test:
        label = path.split('/')[-2]
        gt[path] = label

    #stats
    #print classes
    print "NUMBER OF CLASSES:", len(test_classes)
    print "NUMBER OF TEST SAMPLES:", len(test)

    return test, gt  
    
TEST, GT = parseTestSet()

#################### BATCH HANDLING #####################
def getSignalChunk(sig, rate):

    #split signal into chunks
    sig_splits = spectrogram.splitSignal(sig, rate, SPEC_LENGTH, SPEC_OVERLAP)

    #get batch-sized chunks of image paths
    for i in xrange(0, len(sig_splits), BATCH_SIZE):
        yield sig_splits[i:i+BATCH_SIZE]

def getNextSpecBatch(path):

    #open wav file
    (rate, sig) = wave.read(path)

    #change sample rate if needed
    if rate != 44100:
        sig, rate = spectrogram.changeSampleRate(sig, rate)

    #fill batches
    for sig_chunk in getSignalChunk(sig, rate):

        #allocate numpy arrays for image data and targets
        s_b = np.zeros((BATCH_SIZE, IM_DIM, IM_SIZE[1], IM_SIZE[0]), dtype='float32')

        ib = 0
        for s in sig_chunk:
        
            #load spectrogram data from sig
            spec = spectrogram.getSpecFromSignal(s, rate, SPEC_LENGTH)
            
            #reshape spec
            spec = spec.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])

            #pack into batch array
            s_b[ib] = spec
            ib += 1

        #trim to actual size
        s_b = s_b[:ib]

        #yield batch
        yield s_b

####################  MODEL LOAD  ########################
def loadModel(filename):
    print "IMPORTING MODEL...",
    net_filename = MODEL_PATH + filename

    with open(net_filename, 'rb') as f:
        data = pickle.load(f)

    #for evaluation, we want to load the complete model architecture and trained classes
    net = data['net']
    classes = data['classes']
    im_size = data['im_size']
    im_dim = data['im_dim']
    
    print "DONE!"

    return net, classes, im_size, im_dim

################# PREDICTION FUNCTION ####################
def getPredictionFuntion(net):
    net_output = l.get_output(net, deterministic=True)

    print "COMPILING THEANO TEST FUNCTION...",
    start = time.time()
    test_net = theano.function([l.get_all_layers(NET)[0].input_var], net_output, allow_input_downcast=True)
    print "DONE! (", int(time.time() - start), "s )"

    return test_net

################# PREDICTION POOLING ####################
def predictionPooling(p):

    #You can test different prediction pooling strategies here
    #We only use average pooling
    if p.ndim == 2:
        p_pool = np.mean(p, axis=0)
    else:
        p_pool = p

    return p_pool

####################### TESTING #########################
#test model
print "TESTING MODEL..."

#load model
NET, CLASSES, IM_SIZE, IM_DIM = loadModel(filename=TRAINED_MODEL)

#get test function
test_net = getPredictionFuntion(NET)

pr = []
pcnt = 1
ecnt = 0
acc = []
#test every sample from test collection
for path in TEST:    

    #status
    print pcnt, path.replace(TEST_DIR, ''),

    try:

        #make predictions for batches of spectrograms
        predictions = []
        for spec_batch in bg.threadedBatchGenerator(getNextSpecBatch(path)):

            #predict
            p = test_net(spec_batch)

            #stack predictions
            if len(predictions):
                predictions = np.vstack([predictions, p])  
            else:
                predictions = p

        #prediction pooling
        p_pool = predictionPooling(predictions)

        #get class labels for predictions
        p_labels = {}
        for i in range(p_pool.shape[0]):
            p_labels[CLASSES[i]] = p_pool[i]

        #sort by confidence
        p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)[:MAX_PREDICTIONS]

        #calculate avg precision
        for i in range(len(p_sorted)):
            if p_sorted[i][0] == GT[path]:
                pr.append(1.0 / float(i + 1))
                if i + 1 == 1:
                    acc.append(1)
                else:
                    acc.append(0)
                break

        print 'LABEL:', p_sorted[0], 'AVGP:', pr[-1]
    
    except KeyboardInterrupt:
        break
    except:
        print "ERROR"
        #pr.append(0.0)
        traceback.print_exc()
        ecnt += 1
        continue

    pcnt += 1
    
print "TESTING DONE!"
print "ERRORS:", ecnt, "/", pcnt - 1
print "MAP:", np.mean(pr)
print "ACCURACY:", np.mean(acc)

        

    
