import os
import sys
import matplotlib.pyplot as plt
import numpy as np

args = sys.argv
if(args[1] == "usps_train.csv" and args[2] == "usps_test.csv" and float(args[3])):
    learning_factor = float(args[3])
else:
    print("Incorrect files passed, exiting...")
    quit()

u_train_data = os.getcwd() + '/usps_train.csv'
u_test_data = os.getcwd() + '/usps_test.csv'

train_x = np.genfromtxt(u_train_data, usecols=range(256), delimiter=',')
train_y = np.genfromtxt(u_train_data, usecols=(-1),  delimiter=',')
train_x = train_x/255
train_x = np.insert(train_x, 0, 1, axis=1)
weights = np.zeros(train_x.shape[1])

test_x = np.genfromtxt(u_test_data, usecols=range(256), delimiter=',')
test_y = np.genfromtxt(u_test_data, usecols=(-1),  delimiter=',')
test_x = test_x/255
test_x = np.insert(test_x, 0, 1, axis=1)

train_correct = [0]*1000
test_correct = [0]*1000

for i in range(1000):
    for k in range(train_x.shape[0]):
        if np.dot(weights.T, train_x[k]) >= 0. and train_y[k] == 1:
            train_correct[i]+=1
        elif np.dot(weights.T, train_x[k]) < 0. and train_y[k] == 0:
            train_correct[i]+=1
    train_correct[i] = (train_correct[i]/train_x.shape[0])*100

    for k in range(test_x.shape[0]):
        if np.dot(weights.T, test_x[k]) >= 0. and test_y[k] == 1:
            test_correct[i]+=1
        elif np.dot(weights.T, test_x[k]) < 0. and test_y[k] == 0:
            test_correct[i]+=1
    test_correct[i] = (test_correct[i]/test_x.shape[0])*100
    gradient = np.zeros(train_x.shape[1])
    for j in range(train_x.shape[0]):
        y_hat = 1./(1. + np.e**(-1. * np.dot(weights.T, train_x[j])))
        gradient = np.add(gradient,  ((y_hat-train_y[j]) * train_x[j]))
    weights = np.subtract(weights, (learning_factor * gradient))


plt.plot(range(1000), train_correct)
plt.xlabel('number of epochs')
plt.yticks(range(50, 101, 10))
plt.ylabel('accuracy of model on training data')
plt.savefig("TrainingAccuracy.png")
plt.clf()
plt.plot(range(1000), test_correct)
plt.xlabel('number of epochs')
plt.yticks(range(50, 101, 10))
plt.ylabel('accuracy of model on testing data')
plt.savefig("TestingAccuracy.png")
plt.clf()

print("training accuracy: ", train_correct[-1])
print("testing accuracy: ", test_correct[-1])
