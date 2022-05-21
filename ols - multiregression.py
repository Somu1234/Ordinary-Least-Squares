# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 21:00:54 2022

@author: soumy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

#Ordinary Least Squares estimation
def OLS(x, y):
    #B = (inv(X'X))*(X'Y) - OLS
    X = x.T @ x
    X = np.linalg.inv(X)
    B = X @ (x.T @ y)
    return B
 
def plots(x, y, B):
    X = (x.T)[1]
    Y = (x.T)[2]

    plt.subplot(1, 2, 1)
    plt.title('DATA')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    for i in range(len(X)):
        if y[i] == 1:
            plt.plot(X[i], Y[i], 'gX')
        else:
            plt.plot(X[i], Y[i], 'r+')

    #Predicted vector
    Y_pred = list(x @ B)
    #Convert to classes - 0 or 1
    Y_pred = np.array([0 if y < 0.5 else 1 for y in Y_pred])

    #Show metrics
    print("Accuracy : ", metrics.accuracy_score(y, Y_pred))
    print("Precision : ", metrics.precision_score(y, Y_pred))
    print("Recall : ", metrics.recall_score(y, Y_pred))
    print("Confusion Matrix : \n", metrics.confusion_matrix(y, Y_pred))
    
    plt.subplot(1, 2, 2)
    plt.title('PREDICTIONS - OLS')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    for i in range(len(X)):
        if Y_pred[i] == 1:
            plt.plot(X[i], Y[i], 'm+')
        else:
            plt.plot(X[i], Y[i], 'yX')
    plt.show()
 
if __name__ == '__main__':
    df = pd.read_csv("Dataset/admission_marks.csv")
    #Replace Yes and No with 1/0
    df = df.replace({"Result" : {'YES' : 1, 'NO' : 0}})

    x = df.loc[:, df.columns[0:2]]
    x = x.to_numpy()
    #Add 1 feature to X ([1, x, y]) for decision boundary
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    #Get Y
    y = df.loc[:, df.columns[-1]]
    y = y.to_numpy()

    #Estimating weights
    B = OLS(x, y)
    print("Weights :\nb_0 = {} b_1 = {} b_2 = {}\n".format(B[0], B[1], B[2]))

    #Plotting
    plots(x, y, B)
