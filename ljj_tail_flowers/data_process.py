# coding=utf-8
import numpy as np
from ast import literal_eval
import pandas as pd


train_raw_data = np.genfromtxt('./dataset/iris_train.txt',delimiter=',',dtype=None)
test_raw_data = np.genfromtxt('./dataset/iris_test.txt',delimiter=',',dtype=None)

len_train = len(train_raw_data)
len_test = len(test_raw_data)
#label = np.genfromtxt('./dataset/iris_train.txt',delimiter=',',usecols=4,dtype=None)
#data_class = {'Iris-setosa': 0, 'Iris-versicolor': 1,'Iris-virginica':2}


def process():
    train_data = np.zeros((len_train,4))
    test_data = np.zeros((len_test,4))
    train_label = np.zeros(len_train)
    test_label = np.zeros(len_test)

    for i in range(len_train):
        for j in range(4):
            train_data[i][j] = train_raw_data[i][j]


    for i in range(len_test):
        for j in range(4):
            test_data[i][j] = test_raw_data[i][j]

    train_label = [train_raw_data[i][4] for i in range(len_train)]
    test_label = [test_raw_data[i][4] for i in range(len_test)]

    for i in range(len_train):
        if len(train_label[i])==11:
            train_label[i]=0
        elif len(train_label[i])==15:
            train_label[i]=1
        else:
            train_label[i]=2

    for i in range(len_test):
        if len(test_label[i])==11:
            test_label[i]=0
        elif len(test_label[i])==15:
            test_label[i]=1
        else:
            test_label[i]=2

    return train_label,train_data,test_label,test_data
   # data_class = {'Iris-setosa': 0, 'Iris-versicolor': 1,'Iris-virginica':2}
   # train_label = train_label.map(data_class)









