from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def forward(x, W1, b1 ):
    # L0 = tanh_layer(x, W0, b0)
    L0 = x # first layer output, i.e. no hidden layer
    L1 = dot(W1.T, L0) + b1 # linear combination of x's + bias
    output = softmax(L1)
    return L0, L1, output

def df(x, y, W1, b1):
    L0, L1, p = forward(x, W1, b1)  
    dw = np.subtract(p, y) ## element-wise subtraction
    return dot(dw, x) ## fix, include bias GD; also check x vs. dw dimensions match

def f(x, y, W1, b1):
    L0, L1, p = forward(x, W1, b1)  
    C = NLL(p, y)
    return C

def NLL(y, y_):
    return -sum(y_*log(y)) 

def finite(x, y, theta, p, q):
    ### Fix, I just copied from A1
    h = 0.001
    prev_cost = f(x, y, theta)
    deriv = df(x, y, theta)
    _theta = theta
    theta[p, q] += h  # do some sort of add to the specific weight index i, j
    new_cost = f(x, y, theta)

    return float(abs(new_cost - prev_cost)) * 100 / float(prev_cost)