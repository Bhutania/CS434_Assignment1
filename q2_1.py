import matplotlib.pyplot as plt
import numpy as np
import os
import sys

args = sys.argv
if(args[1] == "usps_train.csv" and args[2] == "usps_test.csv"):
    u_train_data = args[1]
    u_test_data = args[2]
else:
    print("Incorrect files passed, exiting...")
    quit()

np.seterr(all='raise')
learning_factor = .01

train_x = np.genfromtxt(u_train_data, usecols=range(256), delimiter=',')
train_y = np.genfromtxt(u_train_data, usecols=(-1),  delimiter=',')
train_x = train_x/255
train_x = np.insert(train_x, 0, 1, axis=1)
weights = np.zeros(train_x.shape[1])


for i in range(1000):
    gradient = np.zeros(train_x.shape[1])
    for j in range(train_x.shape[0]):
        y_hat = 1./(1. + np.e**(-1. * np.dot(weights.T, train_x[j])))
        gradient = np.add(gradient,  ((y_hat-train_y[j]) * train_x[j]))
    weights = np.subtract(weights, (learning_factor * gradient))

correct = 0

for i in range(train_x.shape[0]):
    if np.dot(weights.T, train_x[i]) >= 0. and train_y[i] == 1:
        correct+=1
    elif np.dot(weights.T, train_x[i]) < 0. and train_y[i] == 0:
        correct+=1

print(correct/train_x.shape[0])