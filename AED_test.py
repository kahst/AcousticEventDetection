#!/usr/bin/env python

print "HANDLING IMPORTS..."

import time
import operator
import argparse

import traceback
import numpy as np
import pickle

import theano

from lasagne import random as lasagne_random
from lasagne import layers as l

import AED_spec as spectrogram

print "...DONE!"

######################## CONFIG #########################
#Fixed random seed
RANDOM_SEED = 1337
RANDOM = np.random.RandomState(RANDOM_SEED)
lasagne_random.set_rng(RANDOM)

#Pre-trained model params
MODEL_PATH = 'model/'
TRAINED_MODEL = 'AED_Example_Run_model.pkl'

################### ARGUMENT PARSER #####################
def parse_args():
    
    parser = argparse.ArgumentParser(description='Acoustic Event Classification')
    parser.add_argument('--filenames', dest='filenames', help='paths to sample wav files for testing as list or single string', type=str, default='')
    parser.add_argument('--modelname', dest='modelname', help='name of pre-trained model', type=str, default=None)
    parser.add_argument('--speclength', dest='spec_length', help='spectrogram length in seconds', type=int, default=3)
    parser.add_argument('--overlap', dest='spec_overlap', help='spectrogram overlap in seconds', type=int, default=2)
    parser.add_argument('--results', dest='num_results', help='number of results', type=int, default=5)
    parser.add_argument('--confidence', dest='min_confidence', help='confidence threshold', type=float, default=0.01)

    args = parser.parse_args()    

    #single test file or list of files?
    if isinstance(args.filenames, basestring):
        args.filenames = [args.filenames]

    return args

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

####################### PREDICT #########################
def predict(img):    

    #transpose image if dim=3
    try:
        img = np.transpose(img, (2, 0, 1))
    except:
        pass

    #reshape image
    img = img.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])

    #calling the test function returns the net output
    prediction = TEST_NET(img)[0] 

    return prediction

####################### TESTING #########################
def testFile(path, spec_length, spec_overlap, num_results, confidence_threshold=0.01):

    #time
    start = time.time()
    
    #extract spectrograms from wav-file and process them
    predictions = []
    spec_cnt = 0
    for spec in spectrogram.getMultiSpec(path, seconds=spec_length, overlap=spec_overlap):

        #make prediction
        p = predict(spec)
        spec_cnt += 1

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
        if p_pool[i] >= confidence_threshold:
            p_labels[CLASSES[i]] = p_pool[i]

    #sort by confidence and limit results (None returns all results)
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)[:num_results]

    #take time again
    dur = time.time() - start

    return p_sorted, spec_cnt, dur

#################### EXAMPLE USAGE ######################
if __name__ == "__main__":

    #adjust config
    args = parse_args()

    #load model
    if args.modelname:
        TRAINED_MODEL = args.modelname
    NET, CLASSES, IM_SIZE, IM_DIM = loadModel(TRAINED_MODEL)

    #compile test function
    TEST_NET = getPredictionFuntion(NET)

    #do testing
    for fname in args.filenames:
        print 'TESTING:', fname
        pred, cnt, dur = testFile(fname, args.spec_length, args.spec_overlap, args.num_results, args.min_confidence)    
        print 'TOP PREDICTION(S):'
        for p in pred:
            print '\t', p[0], int(p[1] * 100), '%'
        print 'PREDICTION FOR', cnt, 'SPECS TOOK', int(dur * 1000), 'ms (', int(dur / cnt * 1000) , 'ms/spec', ')', '\n'

