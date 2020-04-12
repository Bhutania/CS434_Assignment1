import numpy as np
import os
import sys

args = sys.argv
if(args[1] == "housing_train.csv" and args[2] == "housing_test.csv"):
    h_training_data = args[1]
    h_test_data = args[2]
else:
    print("Incorrect files passed, exiting...")
    quit()

training_x = np.genfromtxt(h_training_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')
training_y = np.genfromtxt(h_training_data, usecols=(-1),  delimiter=',')
weights = np.dot(np.linalg.inv(np.dot(training_x.T, training_x)), np.dot(training_x.T, training_y))

test_y = np.genfromtxt(h_test_data, usecols=(-1), delimiter=',')
test_x = np.genfromtxt(h_test_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')

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