import numpy as np
import matplotlib.pyplot as plt
import os

h_training_data = 'housing_train.csv'
h_testing_data = 'housing_test.csv'

# Generate the 20 random features 
rand_feat = []
for x in range(20):
    rand_feat.append(np.random.normal())

# Create the training data
raw_train_x = np.genfromtxt(h_training_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',',)
train_Y = np.genfromtxt(h_training_data, usecols=(-1),  delimiter=',')
train_X = np.insert(raw_train_x, 0, 1, axis=1)

# Create the testing data
raw_test_x = np.genfromtxt(h_testing_data, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',',)
test_Y = np.genfromtxt(h_testing_data, usecols=(-1),  delimiter=',')
test_X = np.insert(raw_test_x, 0, 1, axis=1)

train_err_list = []
test_err_list = []

for i in range(10):
    # print('Iteration ' + str(i) + ':')
    
    # Add the random features here
    train_X = np.insert(train_X, 0, rand_feat[i], axis=1)
    train_X = np.insert(train_X, 0, rand_feat[i+10], axis=1)

    W = np.dot(np.linalg.inv(np.dot(train_X.T, train_X)), np.dot(train_X.T, train_Y))

    # Compute the training error
    train_err=0
    for x in range(0, train_Y.size):
        train_err+=(train_Y[x]-np.inner(train_X[x], W))**2

    train_err=(train_err/train_Y.size)
    train_err_list.append(train_err)
    # print("\tTraining Error: " + str(train_err))

    test_X = np.insert(test_X, 0, rand_feat[i], axis=1)
    test_X = np.insert(test_X, 0, rand_feat[i+10], axis=1)

    # Compute the testing error
    test_err = 0    
    for x in range(0, test_Y.size):
        test_err+=(test_Y[x]-np.inner(test_X[x], W))**2

    test_err=(test_err/test_Y.size)
    test_err_list.append(test_err)
    # print("\tTest Error: " + str(test_err))

# Plot the training and testing ASE's vs d
d = [2,4,6,8,10,12,14,16,18,20]
plt.rcParams['font.family'] = ['serif']

plt.plot(d, train_err_list)
plt.xlabel("d")
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.ylabel("Training ASE")
plt.savefig("TrainingASE.png")
plt.clf()

plt.plot(d, test_err_list)
plt.xlabel("d")
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.ylabel("Testing ASE")
plt.savefig("TestingASE.png")
