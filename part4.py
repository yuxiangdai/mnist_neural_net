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
from part3 import df, f, df_bias
from part2 import forward

M = loadmat("mnist_all.mat")

def loadData():
    trainingSet = np.zeros(shape=(0, 28*28))
    trainingLabel = np.zeros(shape=(0, 10))
    testSet = np.zeros(shape=(0, 28*28))
    testLabel = np.zeros(shape=(0, 10))

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
            trainingSet = np.vstack((trainingSet, matrix))
            trainingLabel = np.vstack((trainingLabel, trainLbl))


        testData = M["test" + str(i)]
        testLbl = np.zeros(10)
        testLbl[i] = 1
        for j in range(len(testData)):
            matrix = testData[j]
            matrix = matrix / 255.0
            testSet = np.vstack((testSet, matrix))
            testLabel = np.vstack((testLabel, testLbl))

    np.savez("p4_trainTest.npz", trainingSet=trainingSet, trainingLabel=trainingLabel, testSet=testSet, testLabel=testLabel)

loadData()

snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

init_W = np.dot(W0, W1)
init_b = b1

alpha = 0.00001

npzfile = np.load("p4_trainTest.npz")
x = npzfile["trainingSet"]
y = npzfile["trainingLabel"]
x_test = npzfile["testSet"]
y_test = npzfile["testLabel"]

grad_descent(f, df, init_W, init_b, x, y, x_test, y_test, alpha, max_iter = 1000)

def grad_descent(f, df, init_W, init_b, x, y, x_test, y_test, init_t, alpha, max_iter = 1000):
    EPS = 1e-5
    
    W = init_W.copy()
    b = init_b.copy()
    # prev_t = init_t - 10*EPS
    prev_W = init_W - 10*EPS
    prev_b = init_b - 10*EPS
    costs = []
    trainAccArr = []
    testAccArr = []

    iter = 0
    while norm(W - prev_W) > EPS and norm(b - prev_b) > EPS and iter < max_iter:
        prev_W = W.copy()
        prev_b = b.copy()

        # prev_t = t.copy()
        gradf = df(x, y, W, b)
        gradf_bias = df_bias(x, y, W, b)
        W -= alpha * gradf
        b -= alpha * gradf_bias

        cost = f(x, y, W, b)
        costs.append(cost)

        L0, L1, output = forward(x, W, b)
        trainCorrect = 0
        for i in range(np.shape(y)[0]):
            if argmax(output) == argmax(y[i]):
                trainCorrect += 1

        trainingAccuracy = float(trainCorrect) / float(np.shape(y)[0])
        trainAccArr.append(trainingAccuracy)

        L0, L1, test_output = forward(x_test, W, b)
        testCorrect = 0
        for i in range(np.shape(y_test)[0]):
            if argmax(test_output) == argmax(y_test[i]):
                testCorrect += 1

        testAccuracy = float(testCorrect) / float(np.shape(y_test)[0])
        testAccArr.append(testAccuracy)

        iter += 1

    np.savez("p4_results.npz", costs=costs, testAccArr=testAccArr, trainAccArr=trainAccArr)
    return costs, testAccArr, trainAccArr

# def grad_descent2(W, b, x, y):

#     costs = []


#     cost = f(x, y, W, b)
#     costs.append(cost)

#     L0, L1, output = forward(x, W, b)
#     trainCorrect = 0
#     if argmax(output) == argmax(y):
#         trainCorrect += 1
        

#     return float(trainCorrect) / float(trainCorrect)


#     iter += 1
    
