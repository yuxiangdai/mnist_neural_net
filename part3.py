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
import math
import cPickle

import os
from scipy.io import loadmat
from part2 import softmax


def forward(x, W1, b1):
    # L0 = tanh_layer(x, W0, b0)
    L0 = x # first layer output, i.e. no hidden layer
    L1 = dot(W1.T, x) + b1 # linear combination of x's + bias
    output = softmax(L1)
    return L0, L1, output

def NLL(y, y_):
    return -sum(y_*log(y)) 


M = loadmat("mnist_all.mat")

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
# pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b = snapshot["b1"].reshape((10,1))

W = np.dot(W0, W1)  # Doing this makes output not all np.ones

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    

x = x/255.0 # Normalize x
### TEST CODE for Part 2 stuff

np.random.seed(5)

# W1 = np.random.rand(784, 10)
# b1 = np.random.rando(1, 10)

L0, L1, output = forward(x, W, b) # why is output all ones?? is np.rand broken or sth
#get the index at which the output is the largest
# y = argmax(output) 
y = np.random.rand(10, 1)
y /= y.sum() ## need to sum to one


def df(x, y, W, b1):
    L0, L1, p = forward(x, W, b1)
    dw = np.subtract(p, y) ## element-wise subtraction
    return dot(dw, x.T).T ## calculate softmax gradient and confirm

def df_bias(x, y, W, b1):
    L0, L1, p = forward(x, W, b1)
    dw = np.subtract(p, y) ## element-wise subtraction
    return dw ## check is this makes sense

def f(x, y, W1, b1):
    L0, L1, p = forward(x, W1, b1)
    C = NLL(p, y)
    return C



def finite(x, y, W, b, p, q):
    h = 0.001
    prev_cost = f(x, y, W, b)
    deriv = df(x, y, W, b)
    _W = W
    _W[p, q] += h  # do some sort of add to the specific weight index i, j, W is shape (784, 10)
    new_cost = f(x, y, _W, b)

    # finDiff = float(abs(new_cost - prev_cost)) * 100 / float(prev_cost)
    finDiff = float(new_cost - prev_cost) / float(h)
    return finDiff, deriv[p, q]
    # return float(abs(new_cost - prev_cost)) * 100 / float(prev_cost)

def finite_bias(x, y, W, b, p, q):
    h = 0.001
    prev_cost = f(x, y, W, b)
    deriv = df_bias(x, y, W, b)  # Change this derivative calc for bias
    _b = b
    _b[q] += h  # Add step to bias
    new_cost = f(x, y, W, _b)

    finDiff = float(new_cost - prev_cost) / float(h)
    return finDiff, deriv[q]

### TEST Part 3 Here
C = f(x, y, W, b)  # this cost works I think

grad = df(x, y, W, b)

finite(x, y, W, b, 220, 0)  ## 221st of digit 0



def test_df():
    for p in range(np.shape(W[:,0])[0]):
        for q in range(np.shape(W[0])[0]):
            if grad[p, q]:
                print(finite(x, y, W, b, p, q))  
                
# test_df()  

            
for p in range(np.shape(W[:,0])[0]):
    for q in range(np.shape(W[0])[0]):
        if grad[p, q]:
            print(q, finite_bias(x, y, W, b, p, q))  