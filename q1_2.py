import numpy as np
import os

h_training_data = os.getcwd() + '/housing_train.csv'
h_test_data = os.getcwd() + '/housing_test.csv'

raw_training_x = np.genfromtxt(h_training_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')
training_y = np.genfromtxt(h_training_data, usecols=(-1),  delimiter=',')
training_x = np.insert(raw_training_x, 0, 1, axis=1)
weights = np.dot(np.linalg.inv(np.dot(training_x.T, training_x)), np.dot(training_x.T, training_y))

test_y = np.genfromtxt(h_test_data, usecols=(-1), delimiter=',')
raw_test_x = np.genfromtxt(h_test_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')
test_x = np.insert(raw_test_x, 0, 1, axis=1)

training_err=0

for x in range(0, training_y.size):
    training_err+=(training_y[x]-np.inner(training_x[x], weights))**2

training_err=(training_err/training_y.size)

print(training_err)

testing_err = 0

for x in range(0, test_y.size):
    testing_err+=(test_y[x]-np.inner(test_x[x], weights))**2

testing_err=(testing_err/training_y.size)

print(testing_err)