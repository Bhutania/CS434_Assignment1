import numpy as np
import os

u_train_data = os.getcwd() + 'usps_train.csv'
u_test_data = os.getcwd() + 'usps_test.csv'

train_x = np.genfromtxt(u_train_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')
train_y = np.genfromtxt(u_train_data, usecols=(-1),  delimiter=',')
train_x = np.insert(train_x, 0, 1, axis=1)

for x in range(1000):
    gradient = numpy.zeros(train_y.size)
    for i in range(train_y.size):
        