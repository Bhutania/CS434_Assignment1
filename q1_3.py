import numpy as np
import os

h_test_data = os.getcwd() + '/housing_train.csv'

X = np.genfromtxt(h_test_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',',)
Y = np.genfromtxt(h_test_data, usecols=(-1),  delimiter=',')
W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

err=0

for x in range(0, Y.size):
    err+=(Y[x]-np.inner(X[x], W))**2

err=(err/Y.size)

print(err)