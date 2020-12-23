import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams


# normalize function
def norm(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return np.divide(vec, norm)


# dictionary for the first letters in the x_train (replace I F M with numbers)
Dict = {'M': '1', 'F': '2', 'I': '3'}

# get training files
file_X = open(sys.argv[1], "r")
file_Y = open(sys.argv[2], "r")

# get files reader
file_read = file_X.readlines()
file_labels = file_Y.readlines()

# insert information to arrays
X = [None] * len(file_read)
i = 0

# for every line in the txt file
for line in file_read:
    # replace the first letter with the number from the dictionsry
    first = Dict[line[0]]
    # continue from line[1] till the end
    copy = line[1:]
    copy = first + copy
    copy = copy[:len(copy) - 1]
    # separate by comma every line to float array
    line_arr = np.array(np.fromstring(copy, dtype=float, sep=','))
    X[i] = line_arr
    i = i + 1

X = np.array(X, dtype=np)
y = []
for line1 in file_labels:
    label = int(line1[0])
    y = np.append(y, label)

# define eta as 0.1
eta = 0.1
# size of line
size = len(X[0])
W = np.zeros((3, size))

# perceptron training - w is the final vector
for j in range(10):
    for x1, y1 in zip(X, y):
        x1 = norm(x1)
        y_hat = np.argmax(np.dot(W, x1))
        if y_hat != y1:
            W[int(y1), :] = W[int(y1), :] + eta * x1
            W[int(y_hat), :] = np.subtract(W[int(y_hat), :], np.multiply(eta, x1))
    eta = eta * 47 / 50

# test

# get the test files path
test_file_X = open(sys.argv[3], "r")
test_file_Y = open(sys.argv[4], "r")
# get the test files readers
test_file_read = test_file_X.readlines()
test_file_labels = test_file_Y.readlines()

# extracting the information to arrays
Xtest = [None] * len(test_file_read)
i = 0
for line in test_file_read:
    first = Dict[line[0]]
    copy = line[1:]
    copy = first + copy
    copy = copy[:len(copy) - 1]
    line_arr = np.array(np.fromstring(copy, dtype=float, sep=','))
    Xtest[i] = line_arr
    i += 1
Xtest = np.array(Xtest, dtype=np)

test_y = []
for line in test_file_labels:
    label = int(line[0])
    test_y = np.append(test_y, label)

# perceptron testing - NumOfErrors is the final num of errors
error = 0
for i in range(len(Xtest)):
    x2 = norm(Xtest[i])
    y_hat = np.argmax(np.dot(W, x2))
    if test_y[i] != y_hat:
        error = error + 1
print("perceptron err = ", float(error) / len(Xtest))
