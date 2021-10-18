import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy import  optimize
import random

x_dim = 4
y_dim = 3
w_dim = (y_dim,x_dim)
b_dim = y_dim #3

def softmax(x):
    return np.exp(x)/np.exp(x).sum()

def loss(w,b,x,y):
    return -np.log(softmax(np.dot(w,x)+b)[y])

def grad(w,b,x,y):
    #init
    w_grad = np.zeros(w.shape) #w:(3,4)
    b_grad = np.zeros(b.shape) #b:(3,)
    s = softmax(np.dot(w,x)+b) #s:(3,)
    w_row = w.shape[0]
    w_col = w.shape[1]
    b_col = b.shape[0]

    for i in range(w_row):
        for j in range(w_col):
            w_grad[i][j] = (s[i] -1) * x[j] if y==i else s[i]*x[j]

    for i in range(b_col):
        b_grad = s[i]-1 if y==i else s[i]

    return w_grad,b_grad

def evaluate(w,b,test_data,test_label):
    num = len(test_data)
    results = []
    for i in range(num):
        label_i = np.dot(w,test_data[i])+b
        res = 1 if softmax(label_i).argmax() == test_label[i] else 0
        results.append(res)

    accuracy = np.mean(results)
    return accuracy



def mini_batch(batch_size, step, epochs, train_data, train_label, test_data, test_label):

    accurate_rates = [] #record accuracy for each epoch
    iters_w = [] #record weights for each epoch
    iters_b = [] #record biases for each epoch

    w = np.zeros(w_dim)
    b = np.zeros(b_dim)

    x_batches = np.zeros(((int(train_data.shape[0] / batch_size), batch_size, 4)))
    y_batches = np.zeros(((int(train_data.shape[0] / batch_size), batch_size)))
    batches_num = int(train_data.shape[0] / batch_size)

    for i in range(0, train_data.shape[0], batch_size):
        x_batches[int(i / batch_size)] = train_data[i:i + batch_size]
        y_batches[int(i / batch_size)] = train_label[i:i + batch_size]

    print('Start training...')


    # print(start)
    for epoch in range(epochs):
        for i in range(batches_num):

            w_gradients = np.zeros(w_dim)
            b_gradients = np.zeros(b_dim)

            x_batch, y_batch = x_batches[i], y_batches[i]
            # x_batch: (batch_size,784)
            # y_batch: (1000,)


            # sum for every in one batch, then get mean value
            for j in range(batch_size):
                w_g, b_g = grad(w, b, x_batch[j], y_batch[j])
                w_gradients += w_g
                b_gradients += b_g
            w_gradients /= batch_size
            b_gradients /= batch_size

            w -= step * w_gradients
            b -= step * b_gradients

            if i % 10 == 0:
                accurate_rates.append(evaluate(w, b, test_data, test_label))

            iters_w.append(w.copy())
            iters_b.append(b.copy())

    return w, b, accurate_rates, iters_w, iters_b



