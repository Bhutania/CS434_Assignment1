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
lambdas = sys.argv[3][1:][:-1].split(",")
train_correct = [0]*len(lambdas)
test_correct = [0]*len(lambdas)

train_x = np.genfromtxt(u_train_data, usecols=range(256), delimiter=',')
train_y = np.genfromtxt(u_train_data, usecols=(-1),  delimiter=',')
train_x = train_x/255
train_x = np.insert(train_x, 0, 1, axis=1)
test_x = np.genfromtxt(u_test_data, usecols=range(256), delimiter=',')
test_x = np.insert(test_x, 0, 1, axis=1)
test_y = np.genfromtxt(u_test_data, usecols=(-1),  delimiter=',')
weights = np.zeros(train_x.shape[1])

for k in range(len(lambdas)):
    # Create the gradient and weight vector
    for i in range(1000):
        gradient = np.zeros(train_x.shape[1])
        for j in range(train_x.shape[0]):
            y_hat = 1./(1. + np.e**(-1. * np.dot(weights.T, train_x[j])))
            gradient = np.add(gradient,  ((y_hat-train_y[j]) * train_x[j]))
        gradient = np.add(gradient, (learning_factor)*(np.linalg.norm(weights)))
        weights = np.subtract(weights, (learning_factor * gradient))

    # Get the number of correct predictions for the training data for kth lambda
    for j in range(train_x.shape[0]):
        if np.dot(weights.T, train_x[j]) >= 0. and train_y[j] == 1:
            train_correct[k]+=1
        elif np.dot(weights.T, train_x[j]) < 0. and train_y[j] == 0:
            train_correct[k]+=1
    train_correct[k] = (train_correct[k]/train_x.shape[0])*100
    print("Training Accuracy for " + str(k) + "lambda: " + str(train_correct[k]))

    # Get the number of correct predictions for the test data for kth lambda
    for j in range(test_x.shape[0]):
        if np.dot(weights.T, test_x[j]) >= 0. and test_y[j] == 1:
            test_correct[k]+=1
        elif np.dot(weights.T, test_x[j]) < 0. and test_y[j] == 0:
            test_correct[k]+=1
    test_correct[k] = (test_correct[k]/test_x.shape[0])*100
    print("Testing Accuracy for " + str(k) + "th lambda: " + str(test_correct[k]))


# Plot the two correct lists vs the given lambdas
plt.plot(lambdas, train_correct)
plt.xlabel("lambdas")
plt.xticks(lambdas)
plt.ylabel("Training Accuracy")
plt.savefig("TrainingAcc.png")
plt.clf()

plt.plot(lambdas, test_correct)
plt.xlabel("lambdas")
plt.xticks(lambdas)
plt.ylabel("Testing Accuracy")
plt.savefig("TestingAcc.png")