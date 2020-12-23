# Natalie balulu ID 312495567
# Batel   Cohen  ID 208521195


import sys
import numpy as np
import random
import numpy.core.defchararray as np_r
from scipy.stats import mstats



# normalize the vector
def norm(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return np.divide(vec, norm)


# testing for algo
def test(x, y, w, name):
    M = 0
    for j in range(0, len(y)):
        y_hat = np.argmax(np.dot(w, x[j]))
        if y[j] != y_hat:
            M = M + 1
    print(name, "err = ", float(M) / len(x))


# the first algorithm - perceptron
def perceptron(data_x, data_y):
    eta = 0.1
    W = np.zeros((3, len(data_x[0])))
    for j in range(100):
        x1 = norm(data_x)
        for x1, y1 in zip(data_x, data_y):
            y_hat = np.argmax(np.dot(W, x1))
            if y_hat != y1:
                W[int(y1), :] = W[int(y1), :] + eta * x1
                W[int(y_hat), :] = np.subtract(W[int(y_hat), :], np.multiply(eta, x1))
        if j > 50:
            eta = eta / 100
    return W


def passive_aggressive(data_x, data_y):
    weight = 0
    counter = 0
    w_pa = np.zeros((3, len(data_x[0])))
    for i in range(10):
        for x, y in zip(data_x, data_y):
            # predict
            y_hat = np.argmax(np.dot(w_pa, x))
            # update
            if y != y_hat:
                loss = max(0, 1 - np.dot(w_pa[y, :], x) + np.dot(w_pa[y_hat, :], x))
                divide = (np.power(np.linalg.norm(x, ord=2), 2) * 2)
                if divide != 0:
                    tau = loss / divide
                    w_pa[y, :] += tau * x
                    w_pa[y_hat, :] -= tau * x
                    counter += 1
                    weight += w_pa
    if counter != 0:
        w_pa = weight / counter
    return w_pa


def svm(data_x, data_y):
    eta = 0.001
    lamda = 0.1
    W = np.zeros((3, len(data_x[0])))
    # normalize data
    for j in range(30):
        c = list(zip(data_x, data_y))
        # random.shuffle(c)
        for x, y in zip(data_x, data_y):
            x = norm(x)
            y_hat = np.argmax(np.dot(W, x))
            if y_hat != y:
                W[int(y), :] = (1 - eta * lamda) * W[int(y), :] + eta * x
                W[int(y_hat), :] = (1 - eta * lamda) * W[int(y_hat), :] - eta * x
        if (j > 10):
            eta = eta / 100
            lamda = lamda / 100
    return W


if __name__ == "__main__":
    # separate the file train_x.txt by comma
    data_x = np.genfromtxt(sys.argv[1], dtype="str", delimiter=",")
    # replace the first letters in the file to the following numbers
    data_x = np_r.replace(data_x, 'M', '0')
    data_x = np_r.replace(data_x, 'F', '1')
    data_x = np_r.replace(data_x, 'I', '2')
    # convert to float np
    data_x = data_x.astype(np.float)
    # separate every line in train_y.txt
    data_y = np.genfromtxt(sys.argv[2], dtype="str", delimiter="\n")
    data_y = data_y.astype(np.float)
    data_y = data_y.astype(np.int)

    # shuffle
    zipM = list(zip(data_x, data_y))
    random.shuffle(zipM)
    data_x, data_y = zip(*zipM)
    # splits into training and test set
    training_x = data_x
    test_x = np.genfromtxt(sys.argv[3], dtype="str", delimiter=",")
    test_x = np_r.replace(test_x, 'M', '0')
    test_x = np_r.replace(test_x, 'F', '1')
    test_x = np_r.replace(test_x, 'I', '2')
    test_x = test_x.astype(np.float)
    training_y = data_y
    # shuffle the training set (again)
    zipM = list(zip(training_x, training_y))
    random.shuffle(zipM)
    training_x, training_y = zip(*zipM)

    w_perceptron = perceptron(training_x, training_y)
    w_pa = passive_aggressive(training_x, training_y)
    w_svm = svm(training_x, training_y)

    for i in range(len(test_x)):
        y_hat_perc = np.argmax(np.dot(w_perceptron, test_x[i]))
        y_hat_svm = np.argmax(np.dot(w_svm, test_x[i]))
        y_hat_pa = np.argmax(np.dot(w_pa, test_x[i]))
        print("perceptron: ",y_hat_perc, ", svm: ", y_hat_svm, ", pa: ", y_hat_pa)

    ##################################################
    ## test
    # w_perceptron1 = perceptron(training_x, training_y)
    # test(test_x, test_y, w_perceptron1, "perceptron")
    # w_pa1 = passive_aggressive(training_x, training_y)
    # test(test_x, test_y, w_pa1, "passive agressive")
    # w_svm1 = svm(training_x, training_y)
    # test(test_x, test_y, w_svm1, "svm")
    ####################################################
