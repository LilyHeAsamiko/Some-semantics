#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:05:33 2020

@author: lilyheasamiko
"""
#classification: 1. not entail 2. entail

import argparse
from gym.spaces import Discrete, Tuple
import logging

#import ray
#from ray import tune
#from ray.tune import function

#from ray.rllib.utils.test_utils import check_learning_achieved

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
from scipy import fftpack, signal

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors
import os
from sklearn import preprocessing
from IPython import embed
import random
import pandas as pd
random.seed(12345)
import re
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle


def WordList(lib=0):
#    if not(isinstance(path)):
    if lib==0:
        pers = ['person', 'people', 'you', 'your', 'yours', 'I', 'me', 'mine', 'he', 'his', 'him', 'she', 'her', 'we', 'us', 'our', 'they', 'them', 'their', 'child', 'children', 'parents', 'boy', 'girl', 'one can', 'anyone', 'woman', 'man']
        things = ['horse', 'airplane', 'camera', 'bridge', 'sidewalk', 'light', 'juice', 'skateboard', 'equipment', 'drinks', 'coffee', 'omelettes', 'water', 'fountain', 'lunch', 'table', 'picture', 'sentence', 'snapshot', 'orange', 'basketball']
        negative = ['not', 'no', 'non', 'none', 'nothing', 'never', 'no','nobody']
        positive = ['is', 'am', 'are', 'does', 'do', 'did', 'done', 'can', 'could','has','have','had','everybody','everyone','all','any']
        loc = ['restaurant', 'home', 'house', 'school', 'hall', 'train', 'street', 'gym', 'road', 'park', 'cross', 'outdoors', 'indoors', 'city', 'country']
        nonloc = ['on', 'in', 'above', 'below', 'north', 'south', 'west', 'east', 'middle', 'up', 'down']    
        exclusive_1 = [ 'he', 'his', 'him', 'boy', 'male','hot','young','stay','keep','fast','true','heavy','fat','out','above','on','up','north','west','far','open','warm']
        exclusive_0 = ['she', 'hers', 'her', 'girl', 'female','cold','old','leave','stop','slow','false','light','skim','in','below','under','down','south','east','near','close','cold']
    Wordlist = [pers, things, negative, positive, loc, nonloc, exclusive_0, exclusive_1]
    return Wordlist

def overlap(l1, l2):
    overlap = 0.0
    for w in l1:
        overlap += np.where(l2 == w, overlap, overlap+1)
    overlap /= np.shape(l1)[0]
    return overlap
        
    
# def LB = labeltf(lb):
#     lowlevel = [] #0: notentail, #1: entail
#    for w in lb:
#     i = 0
#     while (i < np.shape(lb)[0]):
#         if lb[i] == 'entailment':
#             lowlevel.append(1)
#         elif lb[i] == 'neutral':
#             lowlevel.append(2)
#         elif lb[i] == 'contradiction':
#             lowlevel.append(0)
#         else:
 #            print('erroneous label!')
  #       i += 1
#     LB = np.array(lowlevel, dtype = float)
    #assert(np.shape(lowlevel)[0] == np.shape(lb)[0])
#     return LB

def divid(win_size, hop, tf_size=4096): 
    """LSEE-MSTFT algorithm for computing the synthesis window.
    According to: Daniel W. Griffin and Jae S. Lim, `Signal estimation\
    from modified short-time Fourier transform,` IEEE Transactions on\
    Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243,\
    Apr 1984.
    :param window_size: Synthesis window size in samples. 
    :type window_size: int
    :param hop: Hop size in samples.
    :type hop: int
    :param fft_size: FTT size
    :type fft_size: int
    :return: The synthesized window
    :rtype: numpy.core.multiarray.ndarray
    """
    syn_w = signal.hamming(win_size) / np.sqrt(tf_size)
    syn_w_prod = syn_w ** 2
    syn_w_prod.shape = (win_size, 1)
    redundancy = int(win_size / hop)
    env = np.zeros((win_size, 1))

    for k in range(-redundancy, redundancy + 1):
        env_ind = (hop * k)
        win_ind = np.arange(1, win_size + 1)
        env_ind += win_ind

        valid = np.where((env_ind > 0) & (env_ind <= win_size))
        env_ind = env_ind[valid] - 1
        win_ind = win_ind[valid] - 1
        env[env_ind] += syn_w_prod[win_ind]

    syn_w = syn_w / env[:, 0]

    return syn_w


#Hierarchical reward rule: 
def reward2(S1, S2, pers, things):
#Phi1: a1(S1, l1, pers/things)^a2(S2, l2, overlapped catogory)^overlap in (0,0.5]: 1             
    c1 = 0
    c2 = 0
    c11 = 0
    c22 = 0
    for w1 in things:
        c1 += np.sum(S1 == w1)
        c2 += np.sum(S2 == w1)
    for w2 in pers:
        c11 += np.sum(S1 == w2)        
        c22 += np.sum(S2 == w2)
    if c22*c11 >0:
        if c2*c1 > 0:
            return np.exp(1)
        elif c1 +c2 ==0:
            return np.exp(1) 
    elif c11 +c22== 0:
        return np.exp(1)
    return 0

def reward1(S1, S2, negative, positive, exclusive_0, exclisive_1):
#Phi2: a3(S1, l1, negative/positive)^a4(S2, l2, overlapped motion noticing exclusive)^overlap >0: 3
    c1 = 0
    c2 = 0
    c11 = 0
    c22 = 0
    for w1 in negative:
        c1 += np.sum(S1 == w1) 
        c2 += np.sum(S2 == w1)
    if c2 >0 & c1 == 0 : # *, neg
        if mod(c2, 2) == 0 : #nonneg, nonneg
            for w11 in [exclusive_0[:], exclusive_1[:]]:
                c11 += np.sum(S1 == w11)
                c22 += np.sum(S2 == w11)        
                if c11 + c22 ==0:
                    w11 = []
                    return np.exp(3)
                elif c11*c22 > 0:
                    w11 = []
                    return np.exp(3)
        c11 = 0
        c22 = 0
        if mod(c2, 2) >0: #
            for w11 in [exclusive_1[:]]:
                c11 += np.sum(S1 == w11)
                c22 += np.sum(S2 == w11)        
                if c11 >0 & c22 ==0 & (S2 == exclusive_0[exclusive_1==w11]) >0:
                    w11 = []
                    return np.exp(3)
                elif c11*c22 > 0 & (S1 == exclusive_0[exclusive_1==w11]) >0:
                    w11 = []
                    return np.exp(3)
        c1 = 0
        c2 = 0
    if c1 + c2 == 0: # nonneg, nonneg
        for w11 in [exclusive_0[:], exclusive_1[:]]:
            c11 += np.sum(S1 == w11)
            c22 += np.sum(S2 == w11)        
            if c11 + c22 ==0:
                w11 = []
                return np.exp(3)
            elif c11*c22 > 0:
                w11 = []
                return np.exp(3)
    if c1 >0 & c2 == 0 : # neg, *
        if mod(c1, 2) == 0 : #nonneg, nonneg
            for w11 in [exclusive_0[:], exclusive_1[:]]:
                c11 += np.sum(S1 == w11)
                c22 += np.sum(S2 == w11)        
                if c11 + c22 ==0:
                    w11 = []
                    return np.exp(3)
                elif c11*c22 > 0:
                    w11 = []
                    return np.exp(3)
        c11 = 0
        c22 = 0
        if mod(c1, 2) >0: #
            for w11 in [exclusive_1[:]]:
                c11 += np.sum(S1 == w11)
                c22 += np.sum(S2 == w11)        
                if c22 >0 & c11 ==0 & (S1 == exclusive_0[exclusive_1==w11]) >0:
                    return np.exp(3)
                elif c11*c22 > 0 & (S2 == exclusive_0[exclusive_1==w11]) >0:
                    return np.exp(3)   
    return 0

def reward3(S1, S2, loc, nonloc):
#Phi3: a5(S1, l1, not loc)^a6(S2, l2, non loc)^overlap in (0,0.5]: 0.25
    c1 = 0
    c2 = 0
    c11 = 0
    c22 = 0
    for w1 in nonloc:
        c1 += np.sum(S1 == w1)
        c2 += np.sum(S2 == w1)
        for w2 in loc:
            c11 += np.sum(S1 == w2)
            c22 += np.sum(S2 == w2)            
        if c1*c2*c11*c22 >0:
            return np.exp(0.25)
        elif c1 + c2+ c11 + c22 == 0:
            return np.exp(0.25)
        elif c1*c2*(c1+c2)>0: 
            return np.exp(0.25)
        elif c11*c22*(c11+c22)>0:
            return np.exp(0.25)
    return 0

def MarkovProbMatrix(R1, R2, R3, PState, D):
#        R3 (P2)|-------/<-R1,R3(P4)    #According to the 8 State tf 
#    S0--------->S3<-/------>S1         rate and the three basic reward, the 
#  (P1/P5)       ^  R3,R1(P3)/           markov matrix is constructed.
#      \         |          /
#       \        |         /
#     R2 \   (P7)|R2,R3   /
#         \      |       /
#          \     |      /
#           \    |     /R1,R2,R3
#           _\|  |   \/__(P8)
#        (P6) -> S2-^/
#             |     |
#             |- R2-|             
################################################################
    N = np.shape(PState)[0]
    M = np.zeros((N, N))
    Np = 2**(N-1)
    p = np.ones((Np,1))
    if D == 0:
        R1 = np.exp(3)
        R2 = np.exp(1)
        R3 = np.exp(0.25)
    p[0] = np.exp(0.25)
    p[4] = p[0.25]
    p[5] = R2
    p[7] = R2*R1*R3
    p[6] = R2*R3
    p[2] = R3*R1
    p[3] = p[2]
    p[1] = R3
    p = p/np.sum(p)
    M[0,0] =p[0]+p[4]
    M[0,2] =p[5]
    M[0,3] =p[1]
    M[1,2] =p[7]
    M[1,3] =p[3]
    M[2,2] =p[5]
    M[2,3] =p[7]
    M[3,1] =p[3]
    M[3,3] =p[1]
    return [M, p]
     
def generate_data(N, M):

    K1 = np.reshape(np.random.randn(2, N),np.shape(M))
    K2 = np.reshape(np.random.randn(2, N),np.shape(M))

    T1 = np.array([-1, 1]).reshape((2, 1))
    T2 = np.array([-5, 2]).reshape((2, 1))

    X1 = np.dot(M, K1) + np.tile(T1, [1, N]).reshape((4, 4))
    X2 = np.dot(M, K2) + np.tile(T2, [1, N]).reshape((4, 4))

    X1 = X1[::-1,:]
    X2 = X2[::-1,:]

    return X1, X2

     
def grad(w, X, y):
    G = 0
    for n in range(X.shape[0]):
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))
        G += numerator / denominator
    return G

def loss(w, X, y):
    w = np.array(w)
    if len(w.shape) == 2:
        Y = np.tile(y, np.array(w.T.shape[1], 1)).T
    else:
        Y = y
    L = np.sum(1 + np.exp(-Y * np.dot(X, w.T)), axis = 0)
    return L

def log_loss(w, X, y): #1
    w = np.array(w)
    if len(w.shape) == 2:
        Y = np.tile(y, np.array(w).T.shape[1]).T
    else:
        Y = y
    L = np.sum(np.log(1 + np.exp(-(Y).reshape(np.shape(X)) * np.dot(X, np.array(w).T))), axis = 0)
    return L
      
def TotalR(Pstate, S1, S2, D, pers, things, negative, positive, loc, nonloc, exclusive_0, exclusive_1):
    R1 = reward1(S1, S2, negative, positive, exclusive_0, exclusive_1)    
    R2 = reward2(S1, S2, pers, things)
    R3 = reward2(S1, S2, loc, nonloc)
    #Pstate = [1, 1, 1, 1]
    [M, p] = MarkovProbMatrix(R1, R2, R3, Pstate, D)
    return [M, p]
    
def get_model(in_data, out_data):
    """
    Keras model definition
    
    :param in_data: input data to the network (training data)
    :param out_data: output data to the network (training labels)
    :return: _model: keras model configuration
    """
    mel_start = Input(shape=(in_data.shape[-2], in_data.shape[-1]))
    mel_x = GRU(32, dropout=0.24, return_sequences=True)(mel_start)
    mel_x = TimeDistributed(Dense(out_data.shape[-1]))(mel_x)
    out = Activation('sigmoid')(mel_x)

    _model = Model(inputs=mel_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy')
    _model.summary()
    return _model

def load_feat(_input_feat_name, _nb_frames):
    # Load normalized features and pre-process them - splitting into sequence
    dmp = np.load(_input_feat_name)
    train_data, train_labels, test_data, test_labels = \
        split_in_seqs(dmp['arr_0'], _nb_frames), \
        split_in_seqs(dmp['arr_1'], _nb_frames), \
        split_in_seqs(dmp['arr_2'], _nb_frames), \
        dmp['arr_3']
    test_labels_recording = split_in_seqs(dmp['arr_3'], nb_frames)[:, 0]
    return train_data, train_labels, test_data, test_labels, test_labels_recording

def train(_window_length, _nb_mel_bands, _nb_frames):
    # Initialize filenames
#    input_feat_name, model_name, results_csv_name, test_filelist_name = \
#        get_input_output_file_names(_window_length, _nb_mel_bands, _nb_frames)
    input_feat_name = 'entailment_vs_notentailment_{}_{}_{}.npz'.format(_nb_frames, _nb_mel_bands, _window_length)
    model_name = input_feat_name.replace('.npz', '_model.h5')
    test_filelist_name = input_feat_name.replace('.npz', '_filenames.npy')
    results_csv_name = input_feat_name.replace('npz', 'csv')
    print('input_feat_name: {}\n model_name: {}\nresults_csv_name: {}\ntest_filelist_name: {}'.format(
        input_feat_name, model_name, results_csv_name, test_filelist_name))

    
    # Load data
    train_data, train_labels, test_data, test_labels, test_labels_recording = load_feat(input_feat_name, _nb_frames)

    # Load test data file names
    test_filelist = np.load(test_filelist_name)

    # Load the CNN model
    model = get_model(train_data, train_labels)

    nb_epoch = 60      # Maximum number of epochs for training
    batch_size = 16     # Batch size

    patience = int(0.25 * nb_epoch) # We stop training if the accuracy does not improve for 'patience' number of epochs
    patience_cnt = 0    # Variable to keep track of the patience

    best_accuracy = -999    # Variable to save the best accuracy of the model
    best_epoch = -1     # Variable to save the best epoch of the model
    train_loss = [0] * nb_epoch  # Variable to save the training loss of the model per epoch
    framewise_test_accuracy = [0] * nb_epoch  # Variable to save the training accuracy of the model per epoch
    recording_test_accuracy = [0] * nb_epoch  # Variable to save the training accuracy of the model per epoch

    # Training begins
    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')

        # Fit model for one epoch
        hist = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=1
        )

        # save the training loss for the epoch
        train_loss[i] = hist.history.get('loss')[-1]

        # Use the trained model on test data
        pred = model.predict(test_data, batch_size=batch_size)

        # Calculate the accuracy on the test data
        framewise_test_accuracy[i] = metrics.accuracy_score(test_labels, pred.reshape(-1) > 0.5)
        recording_pred = np.mean(pred, 1)
        recording_test_accuracy[i] = metrics.accuracy_score(test_labels_recording, recording_pred > 0.5)
        patience_cnt = patience_cnt + 1

        # Check if the test_accuracy for the epoch is better than the best_accuracy
        if framewise_test_accuracy[i] > best_accuracy:
            # Save the best accuracy and its respective epoch
            best_accuracy = framewise_test_accuracy[i]
            best_epoch = i
            patience_cnt = 0

            # Save the best model
            model.save('{}'.format(model_name))

            # Write the results of the best model to a file
            fid = open(results_csv_name, 'w')
            fid.write('{},{},{},{},{}\n'.format(
                'Index', 'Test file name', 'Groundtruth: entailment = 1 and notentailment=0', 'Predictions: entailment = 1 and notentailment=0',
                '% of sentence frames: closer to 1 means mostly entailment and 0 means mostly not entailment'))
            for cnt, test_file in enumerate(test_filelist):
                fid.write('{},{},{},{},{}\n'.format(cnt, test_file, int(test_labels_recording[cnt, 0]), int(recording_pred[cnt, 0] > 0.5), recording_pred[cnt, 0]))
            fid.close()

        print('paragraphwise_accuracy: {}, article_accuracy: {}, best paragraphwise accuracy: {}, best epoch: {}'.format(framewise_test_accuracy[i], recording_test_accuracy[i], best_accuracy, best_epoch))

        # Early stopping, if the test_accuracy does not change for 'patience' number of epochs then we quit training
        if patience_cnt > patience:
            break

    print('The best_epoch: {} with best paragraphwise accuracy: {}'.format(best_epoch, best_accuracy))


def test(_window_length, _nb_mel_bands, _nb_frames, test_file_index):
    # Initialize filenames
    input_feat_name, model_name, results_csv_name, test_filelist_name = \
        get_input_output_file_names(_window_length, _nb_mel_bands, _nb_frames)

    # Load data
    train_data, train_labels, test_data, test_labels, test_labels_recording = load_feat(input_feat_name, _nb_frames)

    # Load test data file names
    test_filelist = np.load(test_filelist_name)

    # Load trained model
    model = load_model(model_name)

    # Choose the feature for input file and format it to right dimension
    test_feat = test_data[test_file_index][np.newaxis]

    # predict the class using trainedmodel
    pred = model.predict(test_feat)
    pred = np.squeeze(pred)

    # Load audio file to extract spectrogram. This is done here only to visualize the audio.
    if test_file_index < len(test_filelist)/2:
        test_audio_filename = os.path.join(__music_audio_folder, test_filelist[test_file_index])
    else:
        test_audio_filename = os.path.join(__speech_audio_folder, test_filelist[test_file_index])

    y, sr = librosa.load(test_audio_filename)
    stft = librosa.stft(y, n_fft=_window_length, hop_length=_window_length//2, win_length=_window_length)
    stft = np.abs(stft[:500, :nb_frames])**2  # visualizing only the first 500 bins and nb_frames

    # Visualize the spectrogram and model outputs
    time_vec = np.arange(stft.shape[1])*_window_length/(2.0*sr)
    plot.figure()
    plot.subplot(211), plot.pcolormesh(time_vec, range(stft.shape[0]), stft, norm=colors.PowerNorm(0.25, vmin=stft.min(), vmax=stft.max())), plot.title('Spectrogram for {}'.format(test_filelist[test_file_index]))
    plot.xlabel('Time'), plot.ylabel('Spectral bins')
    plot.subplot(212), plot.hold(True), plot.plot(time_vec, pred, label='RNN output'), plot.plot(time_vec, 0.5*np.ones(pred.shape), label='Threshold'), plot.hold(False)
    plot.xlabel('Time'), plot.ylabel('RNN output magnitude')
    plot.grid(True), plot.ylim([-0.1, 1.2]), plot.title('CNN model output')
    plot.legend()
    plot.show()


# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------

#os.path(r'Users/he/Downloads/Arena')
#DIR_PATH = os.path.dirname(os.path.realpath(r'Downloads/Arena/'))
#data = np.loadtxt({DIR_PATH}/'preproc1_expl_1.train.txt',delimiter = {' ','/n'});
#DIR_PATH = os.path.dirname(os.path.realpath(r'D:/PhD in Oxford,Ethz,KI,others/OxfordPhD/Arena/'))
#data = np.loadtxt({DIR_PATH+'preproc1_expl_1.train.txt'},delimiter = ' ');
data = pd.read_csv(r'Downloads/Arena/preproc1_expl_1.train.txt', sep = "\.\;\/n", nrows = 1000, dtype = str, engine = 'python')
label = pd.read_csv(r'D:/PhD in Oxford,Ethz,KI,others/OxfordPhD/Arena/labels.train.txt', sep = "\.\;\/n", nrows = 1000, dtype = str, engine = 'python')

S1 = []
S2 = [] 
LB1 =[]
LB2 = []
LB = []
#LB = labeltf(label)
i = 0
#for i in range(0,2,np.shape(data)[0]):
#for i in np.linspace(0,np.shape(data)[0]-2,int(np.shape(data)[0]/2)):
while (i < np.shape(data)[0]-1):
    temp = str(data.iloc[i,])    
    temp = temp.replace("Name: "+str(i)+", dtype: object",' ')
    temp = temp.replace(',',' ')
    temp = temp.replace('\n',' ')
    temp = temp.replace('...',' ')
    # temp = re.sub('["Name: "+str(string)+"dtype: object",\n...]',' ',temp)
    l1 = temp.split(' ')
    temp = []
#    ntemp = np.shape(l1)[0]
#    N +=  ntemp
    temp = str(data.iloc[i+1,]) 
    temp = temp.replace("Name: "+str(i)+", dtype: object",' ')
    temp = temp.replace(',',' ')
    temp = temp.replace('\n',' ')
    temp = temp.replace('...',' ')
    l2 = temp.split(' ')  
    
    if (np.shape(l1)[0] <= np.shape(l2)[0]) & (overlap(l1, l2) >0):
        S1.append(l1) #for shorter sentences
        S2.append(l2)
#        RewardMat.append()
#        RewardVec.append()
       
        i += 2 
        l1 = []
        l2 = []
        temp = []
    elif (np.shape(l1)[0] > np.shape(l2)[0]) & (overlap(l1, l2) >0):
        S1.append(l2) #for shorter sentences
        S2.append(l1)
        i += 2 
        l1 = []
        l2 = []
        temp = []
    elif overlap(l1, l2) < 0:
        temp = str(data.iloc[i+2,]) 
 #       temp = temp.replace("Name: "+str(i+2)+", dtype: object",' ')
        temp = temp.replace(',',' ')
        temp = temp.replace('\n',' ')
  #      temp = temp.replace('...',' ')
        l3 = temp.split(' ') 
        temp = str(data.iloc[i+3,]) 
   #     temp = temp.replace("Name: "+str(i+3)+", dtype: object",' ')
        temp = temp.replace(',',' ')
        temp = temp.replace('\n',' ')
    #    temp = temp.replace('...',' ')
        l4 = temp.split(' ') 
        if overlap(l1, l3) >0:
            if (overlap(S1[-1], l1)< overlap(l1, l3)) & (overlap(l3, l4)< overlap(l1, l3)) &  (overlap(l2, l3)< overlap(l1, l3)):
                if (np.shape(l1)[0] <= np.shape(l3)[0]):                                    
                    S1.append(l1) #for shorter sentences
                    S2.append(l3)
                
 #                   if lb[i+2] == 'entailment':
 #                       LB.append(1)
 #                       LB1.append(1)
 #                   else:
#                        LB.append(0) 
#                        LB1.append(0)
                    
                else:
                    S1.append(l3) #for shorter sentences
                    S2.append(l1)
#                    if lb[i+2] == 'entailment':
 #                       LB.append(1)
#                        LB2.append(1)
#                    else:
#                        LB.append(0) 
#                        LB2.append(0)
#                    
                i += 3 
  #              i += 1
                l1 = []
                l2 = []
                l3 = []
                l4 = []
                temp = []
            elif overlap(l3, l2)> overlap(l1, l3):
                if (np.shape(l2)[0] <= np.shape(l3)[0]):                                    
                    S1.append(l2) #for shorter sentences
                    S2.append(l3)
                else:
                    S1.append(l3) #for shorter sentences
                    S2.append(l2)
                    
#                if lb[i+2] == 'entailment':
#                    LB.append(1)
#                    LB1.append(1)
#                else:
#                    LB.append(0) 
#                    LB1.append(0)
                    
                i += 3
              #  i += 1
                l1 = []
                l2 = []
                l3 = []
                l4 = []
                temp = []
  
[pers, things, negative, positive, loc, nonloc, exclusive_0, exclusive_1] =WordList(lib = 0)


Batch_size = 200
Train_rate = 0.9
Win_size = 8
Hop_size = 4
nb_melbands = 1
nb_frame = 64

if dynamic == 1:
    Stop_size = 0
    
Stop_size = Batch_size*Train_rate
End = np.shape(S1[0])[0]-Batch_size
RewardVec =[]
RewardMat =[]
Train_LB = []
Test_LB =[]
D = 0
TF_size = Stop_size
        #np.shape(S1)[0]           
Step_size = divid(Win_size, Hop_size, TF_size)
i = 1
#for i in range(1,End):
while i < End:
    if Stop_size > 0:  
        #Consider onespecial problem P(R>=0.9[X(~a2 V a1)]):exclude p6, p7 from the p and M
        [M, p] = TotalR([1, 1, 1, 1], S1[i], S2[i], D, pers, things, negative, positive, loc, nonloc, exclusive_0, exclusive_1)
        M1 = M
        M1[0,2] = 0
        M1[2,2] = 0
        M1[2,3] = 0
        p1 = p
        p1[5] = 0
        p1[6] = 0
        Stop_size -= 1
        i += 1
        RewardMat.append(M1)
        RewardVec.append(p1)
        TrainS1_LB.append(lb[i])
        
#generate training data with markov matrix
np.ransom.seed(1)
X1, X2 = generate_data(N, M)

X = np.concatenate((X1.T, X2.T))
y = np.concatenate((np.ones(int(N/2)), -1 * np.ones(int(N/2))))
idx = np.arange(y.size)
random.shuffle(idx)
mx,nx = np.shape(X)
X = X[idx,:]
y = y[idx]

#preprocess
X[:, 0] = X[:, 0] - X[:, 0].mean()
X[:, 1] = X[:, 1] - X[:, 1].mean()

#w = np.array([1, -1])
w = np.concatenate((np.ones(int(N/4)), -1 * np.ones(int(N/4))))

step_size = 0.001
W = []
accuracies = []
losses = []

for iteration in range(100):
    w = w - step_size * grad(w, X, y)

    loss_val = log_loss(w, X, y)
    print (loss_val)
    print ("Iteration %d: w = %s (log-loss = %.2f)" % \
          (iteration, str(w), loss_val))
    # Predict class 1 probability
    y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
    # Threshold at 0.5 (results are 0 and 1)
    y_pred = (y_prob > 0.5).astype(int)
    # Transform [0,1] coding to [-1,1] coding
    y_pred = 2*y_pred - 1

    accuracy = np.mean(y_pred == y)
    accuracies.append(accuracy)
    losses.append(loss)

    W.append(w)

W = np.array(W)

# Plot the path.

fig, ax = plt.subplots(2, 1, figsize = [5, 5])

xmin = -1
xmax = 1.5
ymin = -1
ymax = 4

Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, int(N/2)),
                     np.linspace(ymin, ymax, int(N/2)))
Wg = np.vstack([-Xg.ravel(), -Yg.ravel(), Xg.ravel(), Yg.ravel()]).T
Temp = np.vstack([-Xg.ravel(), -Yg.ravel(), Xg.ravel(), Yg.ravel()])
Wg = Temp.T
Zg = log_loss(np.reshape(Wg[0:0.5*np.shape(Wg)[0]], np.shape(X)), X, y)
Zg = np.reshape(Zg, Xg.shape)
levels = np.linspace(70, 100, 20)

for i in range(np.shape(X)[1]):    
    plt.figure(),
    ax[0].contourf(Xg[:,i], Yg[:,i], Wg, #Zg,
                  alpha=0.5,
                  cmap=plt.cm.bone,
                  levels = levels)
    
    ax[0].plot(W[:, 0], W[:, 1], 'ro-')
    ax[0].set_xlabel('w$_0$')
    ax[0].set_ylabel('w$_1$')
    ax[0].set_title('Optimization path')
    ax[0].grid()
    ax[0].axis([xmin, xmax, ymin, ymax])
    
    ax[0].annotate('Starting point',
             xy=(W[0, 0], W[0, 1]),
             xytext=(-0.5, 0),
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=0.15",
                            shrinkA = 0,
                            shrinkB = 8,
                            fc = "g",
                            ec = "g"),
             horizontalalignment='center',
             verticalalignment='middle')
    
    
    ax[0].annotate('Endpoint',
             xy=(W[-1, 0], W[-1, 1]),
             xytext=(1.2, 0),
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center',
             verticalalignment='middle')

    ax[1].plot(100.0 * np.array(accuracies), linewidth = 2,
               label = "%d Classification Accuracy" % i)
    ax[1].set_ylabel('Accuracy / %')
    ax[1].set_xlabel('Iteration')
    ax[1].legend(loc = 4)
    ax[1].grid()
    plt.tight_layout()
plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")


train(Win_size, nb_mel_bands, nb_frames)
file_index_in_csv_file = 12
test(Win_size, nb_mel_bands, nb_frames, file_index_in_csv_file)

        
                    
#CL

#CLL                   
   
#MCC
                    
        else:
            step_size =1
        if len(S1[0])*len(S2[0]) > 0:            
            [pers, things, negative, positive, loc, nonloc] = WordList(0)
            R1 = reward1(S1[-1], S2[-1], pers, things)
            R2 = reward2(S1[-1], S2[-1], negative, positive)
            R3 = reward3(S1[-1], S2[-1], loc, nonloc)
                
  