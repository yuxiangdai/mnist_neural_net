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
from part2 import softmax, forward

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

M = loadmat("mnist_all.mat")

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
# pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
# W1 = snapshot["W1"]
# b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    

x = x/255.
### TEST CODE for Part 2 stuff

W0 = np.random.random(7840).reshape((784, 10))
b0 = np.random.random(10).reshape((1, 10))
L0, L1, output = forward(x, W0, b0)
#get the index at which the output is the largest
y = argmax(output) ## exoect 




### TEST Part 3 Here
f(x, y, W1, b1)