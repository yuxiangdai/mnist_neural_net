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

Y = []

def loadData():
    # trainingSet = np.zeros(shape=(0, 28*28))
    trainingSet = []
    # trainingLabel = np.zeros(shape=(0, 10))
    trainingLabel = []
    testSet = np.zeros(shape=(0, 28*28))
    testSet = []
    testLabel = np.zeros(shape=(0, 10))
    testLabel = []

    for i in range(10):
        print "Current i-value: ", i
        trainData = M["train" + str(i)]
        trainLbl = np.zeros(10)
        trainLbl[i] = 1
        for j in range(len(trainData)):
            if j % 1000 == 0:
                print j
            matrix = trainData[j]
            matrix = matrix / 255.0
            trainingSet.append(matrix)
            trainingLabel.append(trainLbl)

        testData = M["test" + str(i)]
        testLbl = np.zeros(10)
        testLbl[i] = 1
        for j in range(len(testData)):
            matrix = testData[j]
            matrix = matrix / 255.0
            testSet.append(matrix)
            testLabel.append(testLbl)
        print(np.shape(trainingSet), np.shape(trainingLabel), np.shape(testSet), np.shape(testLabel))
        
    np.savez("p4_trainTest.npz", trainingSet=trainingSet, trainingLabel=trainingLabel, testSet=testSet, testLabel=testLabel)

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

    np.savez("p5_results.npz", costs=costs, testAccArr=testAccArr, trainAccArr=trainAccArr, W=W, b=b)

    return costs, testAccArr, trainAccArr

# loadData()

snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

# init_W = np.dot(W0, W1)
# init_b = b1
init_W = np.zeros((28 * 28, 10))
init_b = np.zeros((10, 1))

alpha = 0.00001

npzfile = np.load("p4_trainTest.npz")
x = npzfile["trainingSet"]
y = npzfile["trainingLabel"].T
x_test = npzfile["testSet"]
y_test = npzfile["testLabel"]

momentum_grad_descent(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 1000, beta=0.9)

results = np.load("p5_results.npz")
costs = results["costs"]
testAccArr = results["testAccArr"]
trainAccArr = results["trainAccArr"]


x = range(len(trainAccArr))

plt.plot(x, trainAccArr, label="Training set")
plt.plot(x, testAccArr, label="Test set")
plt.xlabel('# Iterations')
plt.ylabel('% Accuracy')
plt.legend(loc='upper left')
plt.show()
