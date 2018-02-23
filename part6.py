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

M = loadmat("mnist_all.mat")

def NLL(y, y_):
    return -sum(y_*log(y)) 

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def forward(x, W, b):
    # L0 = tanh_layer(x, W0, b0)
    L0 = x # first layer output, i.e. no hidden layer
    L1 = dot(W.T, x.T) + b # linear combination of x's + bias
    output = softmax(L1)
    return output

def df(x, y, W, b):
    p = forward(x, W, b)
    dw = np.subtract(p, y) ## element-wise subtraction
    return dot(dw, x).T ## calculate softmax gradient and confirm

def df_bias(x, y, W, b):
    p = forward(x, W, b)
    dw = np.subtract(p, y) ## element-wise subtraction
    bias = dw.sum(axis=1) 
    return bias.reshape(10, 1) ## check is this makes sense

def f(x, y, W, b):
    p = forward(x, W, b)
    C = NLL(p, y)
    return C

def momentum_grad_descent(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 1000, beta=0.9):
    EPS = 1e-5
    
    W = init_W.copy()
    b = init_b.copy()
    # prev_t = init_t - 10*EPS
    prev_W = init_W - 10*EPS
    prev_b = init_b - 10*EPS
    costs = []
    trainAccArr = []
    testAccArr = []
    prev_gradf = 0.0
    prev_gradf_bias = 0.0

    iter = 0
    while norm(W - prev_W) > EPS and norm(b - prev_b) > EPS and iter < max_iter:
        prev_W = W.copy()
        prev_b = b.copy()

        # prev_t = t.copy()
        gradf = df(x, y, W, b)
        gradf_bias = df_bias(x, y, W, b)
        W -= alpha * (beta * prev_gradf + gradf)
        b -= alpha * (beta * prev_gradf_bias + gradf_bias)
        prev_gradf = gradf.copy()
        prev_gradf_bias = gradf_bias.copy()

        cost = f(x, y, W, b)
        costs.append(cost)

        output = forward(x, W, b)
        trainCorrect = 0
        for i in range(np.shape(y)[1]):
            if argmax(output[:, i]) == argmax(y[:, i]):
                trainCorrect += 1

        trainingAccuracy = float(trainCorrect) / float(np.shape(y)[1])
        trainAccArr.append(trainingAccuracy)

        test_output = forward(x_test, W, b)
        testCorrect = 0
        for i in range(np.shape(y_test)[0]):
            if argmax(test_output[:, i]) == argmax(y_test[i, :]):
                testCorrect += 1

        testAccuracy = float(testCorrect) / float(np.shape(y_test)[0])
        testAccArr.append(testAccuracy)

        iter += 1
        if iter % 100 == 0:
            print iter

    np.savez("p6_preproc.npz", W=W, b=b)

    return costs, testAccArr, trainAccArr

def contourPlot(W, b, x, y):
    '''
    Code for Part 6a   
    '''

    # number of data points for contour plotting
    diff = 0.05 # difference between data points tested

    w1 = W[403, 9]
    w2 = W[325, 8]
    w1s = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2s = np.arange(w2 - 0.1, w2 + 0.1, 0.01)

    w1z, w2z = np.meshgrid(w1s, w2s)
    
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[403, 9] =  w1
            W[325, 8] =  w2
            C[i,j] = f(x, y, W, b)
        print i



    plt.figure()
    CS = plt.contour(w1z, w2z, C)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot: Part 6a')
    plt.show()
    
    np.savez("p6_a.npz", w1z=w1z, w2z=w2z, C=C)

def two_weight_gd(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 10, beta=0.9):
    EPS = 1e-5
    
    W = init_W.copy()
    b = init_b.copy()
    prev_W = init_W - 10*EPS
    prev_b = init_b - 10*EPS
    costs = []
    trainAccArr = []
    testAccArr = []
    w1w2 = []
  
    iter = 0
    while norm(W - prev_W) > EPS and norm(b - prev_b) > EPS and iter < max_iter:
        prev_W = W.copy()
        prev_b = b.copy()
        w1w2.append((W[403, 9], W[325, 8]))
        # prev_t = t.copy()
        gradf = df(x, y, W, b)
        gradf_bias = df_bias(x, y, W, b)

        W[403, 9] -= alpha * gradf[403, 9]
        b[9] -= alpha * gradf_bias[9]
        W[325, 8] -= alpha * gradf[325, 8]
        b[8] -= alpha *  gradf_bias[8]

        iter += 1
        print iter

    return w1w2

    np.savez("p6_b.npz", w1w2 = w1w2)

def momentum_two_weight_gd(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 20, beta=0.9):
    EPS = 1e-5
    
    W = init_W.copy()
    b = init_b.copy()
    # prev_t = init_t - 10*EPS
    prev_W = init_W - 10*EPS
    prev_b = init_b - 10*EPS
    costs = []
    trainAccArr = []
    testAccArr = []
    prev_gradf = np.zeros((784,10))
    prev_gradf_bias = np.zeros((10,1))
    w1w2 = []
  
    iter = 0
    while norm(W - prev_W) > EPS and norm(b - prev_b) > EPS and iter < max_iter:
        prev_W = W.copy()
        prev_b = b.copy()
        w1w2.append((W[403, 9], W[325, 8]))
        # prev_t = t.copy()
        gradf = df(x, y, W, b)
        gradf_bias = df_bias(x, y, W, b)

        W[403, 9] -= alpha * (beta * prev_gradf[403, 9] + gradf[403, 9])
        b[9] -= alpha * (beta * prev_gradf_bias[9] + gradf_bias[9])
        W[325, 8] -= alpha * (beta * prev_gradf[325, 8] + gradf[325, 8])
        b[8] -= alpha * (beta * prev_gradf_bias[8] + gradf_bias[8])

        prev_gradf = gradf.copy()
        prev_gradf_bias = gradf_bias.copy()



        iter += 1
        print iter

    return w1w2

    np.savez("p6_c.npz", w1w2=w1w2)


def part6a():
    snapshot = cPickle.load(open("snapshot50.pkl"))

    init_W = np.zeros((28 * 28, 10))
    init_b = np.zeros((10, 1))

    alpha = 0.00001

    npzfile = np.load("p4_trainTest.npz")
    x = npzfile["trainingSet"]
    y = npzfile["trainingLabel"].T
    x_test = npzfile["testSet"]
    y_test = npzfile["testLabel"]

    ## Get Data from p5
    # momentum_grad_descent(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 1000, beta=0.9)

    results = np.load("p6_preproc.npz")
    W = results["W"]
    b = results["b"]

    contourPlot(W, b, x, y)


def part6b():
    npzfile = np.load("p4_trainTest.npz")
    x = npzfile["trainingSet"]
    y = npzfile["trainingLabel"].T
    x_test = npzfile["testSet"]
    y_test = npzfile["testLabel"]
    alpha = 0.001

    results = np.load("p6_preproc.npz")
    W = results["W"]
    b = results["b"]

    # 0.4
    _W = W.copy()
    _W[403, 9] += 0.1
    _W[325, 8] += 0.1

    gd_traj = two_weight_gd(f, df, _W, b, x, y, x_test, y_test, alpha, max_iter = 20, beta=0.9)
    
    w1 = W[403, 9]
    w2 = W[325, 8]
    w1s = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2s = np.arange(w2 - 0.1, w2 + 0.1, 0.01)

    w1z, w2z = np.meshgrid(w1s, w2s)
    
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[403, 9] =  w1
            W[325, 8] =  w2
            C[i,j] = f(x, y, W, b)
        print i

    plt.figure()
    CS = plt.contour(w1z, w2z, C)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    # plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.title('6b: No Momentum')
    plt.show()


def part6c():
        

    npzfile = np.load("p4_trainTest.npz")
    x = npzfile["trainingSet"]
    y = npzfile["trainingLabel"].T
    x_test = npzfile["testSet"]
    y_test = npzfile["testLabel"]
    alpha = 0.0001 # 0.0001 is good, 0.001 is not

    results = np.load("p6_preproc.npz")
    W = results["W"]
    b = results["b"]

    _W = W.copy()
    _W[403, 9] += 0.1
    _W[325, 8] += 0.1

    mo_traj = momentum_two_weight_gd(f, df, _W, b, x, y, x_test, y_test, alpha, max_iter = 20, beta=0.9)
    
    w1 = W[403, 9]
    w2 = W[325, 8]
    w1s = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2s = np.arange(w2 - 0.1, w2 + 0.1, 0.01)

    w1z, w2z = np.meshgrid(w1s, w2s)
    
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[403, 9] =  w1
            W[325, 8] =  w2
            C[i,j] = f(x, y, W, b)
        print i

    plt.figure()
    CS = plt.contour(w1z, w2z, C)
    # plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.title('6c: Momentum')
    plt.show()


def part6e():
    
    npzfile = np.load("p4_trainTest.npz")
    x = npzfile["trainingSet"]
    y = npzfile["trainingLabel"].T
    x_test = npzfile["testSet"]
    y_test = npzfile["testLabel"]
    alpha = 0.0001 # 0.0001 is good, 0.001 is not

    results = np.load("p6_preproc.npz")
    W = results["W"]
    b = results["b"]
    _W = W.copy()
  
    # 0.4
    _W[403, 9] += 0.1
    _W[325, 8] += 0.1

    mo_traj = momentum_two_weight_gd(f, df, _W, b, x, y, x_test, y_test, alpha, max_iter = 20, beta=0.1)
    gd_traj = two_weight_gd(f, df, _W, b, x, y, x_test, y_test, alpha, max_iter = 20)
    
    w1 = W[403, 9]
    w2 = W[325, 8]
    w1s = np.arange(w1 - 0.1, w1 + 0.1, 0.01)
    w2s = np.arange(w2 - 0.1, w2 + 0.1, 0.01)

    w1z, w2z = np.meshgrid(w1s, w2s)
    
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[403, 9] =  w1
            W[325, 8] =  w2
            C[i,j] = f(x, y, W, b)
        print i

    plt.figure()
    CS = plt.contour(w1z, w2z, C)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.title('6e: Momentum Low Beta')
    plt.legend(loc='upper left')
    plt.show()

# part6a()
part6e()