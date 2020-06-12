#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:05:33 2020

@author: he
"""
#classification: 1. not entail 2. entail
#rule: 
#Phi1: a1(S1, l1, pers)^a2(S2, l2, things)^overlap<0.5:3
#Phi2: a3(S1, l1, negative)^a4(S2, l2, positive)^overlap<0.5:1
#Phi3: a4(S1, l1, loc)^a6(S3, l2, nonloc)^overlap<0.5:0.25

import argparse
from gym.spaces import Discrete, Tuple
import logging

import ray
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

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors
import os
from sklearn import preprocessing
from IPython import embed
import random
random.seed(12345)

pers = ['person', 'people', 'you', 'your', 'yours', 'I', 'me', 'mine', 'he', 'his', 'him', 'she', 'her', 'we', 'us', 'our', 'they', 'them', 'their', 'child', 'children', 'parents', 'boy', 'girl', 'one can', 'anyone', 'woman', 'man'];
things = ['horse', 'airplane', 'camera', 'bridge', 'sidewalk', 'light', 'juice', 'skateboard', 'equipment', 'drinks', 'coffee', 'omelettes', 'water', 'fountain', 'lunch', 'table', 'picture', 'sentence', 'snapshot', 'orange', 'basketball'];
negative = ['not', 'no', 'non', 'none', 'nothing', 'never', 'no'];
poaitive = ['is', 'am', 'are', 'does', 'do', 'did', 'done', 'can', 'could'];
loc = ['restaurant', 'home', 'house', 'school', 'hall', 'train', 'street', 'gym', 'road', 'park', 'cross', 'outdoors', 'indoors', 'city', 'country'];
nonloc = ['on', 'in', 'above', 'below', 'north', 'south', 'west', 'east', 'middle', 'up', 'down'];         


# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------

#os.path(r'Users/he/Downloads/Arena')
DIR_PATH = os.path.dirname(os.path.realpath(r'Downloads/Arena/'))
#data = np.loadtxt({DIR_PATH}/'preproc1_expl_1.train.txt',delimiter = {' ','/n'});
data = np.loadtxt(r'Users/he/Downloads/Arena/preproc1_expl_1.train.txt',delimiter = {' ','/n'});

open()




