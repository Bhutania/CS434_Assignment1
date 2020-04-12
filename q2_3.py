import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# args = sys.argv
# if(args[1] == "usps_train.csv" and args[2] == "usps_test.csv"):
# u_train_data = args[1]
u_train_data = "usps_train.csv"
# u_test_data = args[2]
u_test_data = "usps_test.csv"
# else:
#     print("Incorrect files passed, exiting...")
#     quit()

np.seterr(all='raise')
learning_factor = .01
lambdas = sys.argv[3][1:len(sys.argv[3])-1].split(",")
train_list = []
test_list = []

train_x = np.genfromtxt(u_train_data, usecols=range(256), delimiter=',')
train_y = np.genfromtxt(u_train_data, usecols=(-1),  delimiter=',')
train_x = train_x/255
train_x = np.insert(train_x, 0, 1, axis=1)
test_x = np.genfromtxt(u_test_data, usecols=range(256), delimiter=',')
test_x = np.insert(test_x, 0, 1, axis=1)
test_y = np.genfromtxt(u_test_data, usecols=(-1),  delimiter=',')
weights = np.zeros(train_x.shape[1])

for k in lambdas:
    for i in range(1000):
        gradient = np.zeros(train_x.shape[1])
        for j in range(train_x.shape[0]):
            y_hat = 1./(1. + np.e**(-1. * np.dot(weights.T, train_x[j])))
            gradient = np.add(gradient,  ((y_hat-train_y[j]) * train_x[j]))
        gradient = np.add(gradient, (learning_factor)*(np.linalg.norm(weights)))
        weights = np.subtract(weights, (learning_factor * gradient))
    
    # Get the training error for each weight vector
    train_err=0
    for x in range(0, train_y.size):
        train_err+=(train_y[x]-np.inner(train_x[x], weights))**2
    train_err=(train_err/train_y.size)
    print("train: " + str(train_err))
    # Keep track of them
    train_list.append(train_err)

    # Get the testing error for each weight vector
    test_err=0
    for x in range(0, test_y.size):
        test_err+=(test_y[x]-np.inner(test_x[x], weights))**2
    test_err=(test_err/test_y.size)
    print("test: " + str(test_err))
    # Keep track of them
    test_list.append(test_err)

correct = 0

for i in range(train_x.shape[0]):
    if np.dot(weights.T, train_x[i]) >= 0. and train_y[i] == 1:
        correct+=1
    elif np.dot(weights.T, train_x[i]) < 0. and train_y[i] == 0:
        correct+=1

print(correct/train_x.shape[0])

plt.plot(lambdas, train_list)
plt.xlabel("lambdas")
plt.xticks(lambdas)
plt.ylabel("Training Accuracy")
plt.savefig("TrainingAcc.png")
plt.clf()

plt.plot(lambdas, test_list)
plt.xlabel("lambdas")
plt.xticks(lambdas)
plt.ylabel("Testing Accuracy")
plt.savefig("TestingAcc.png")