import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data/train.csv")
data = np.array(data)
m, n = data.shape # m is the number of observations, n is the number of features + 1 (label)
np.random.shuffle(data)

vldData = data[0:1000].T
vldLabels = vldData[0]
vldInputs = vldData[1:n]
vldInputs = vldInputs/255.

trnData = data[1000:m].T
trnLabels = trnData[0]
trnInputs = trnData[1:n]
trnInputs = trnInputs/255.

def init_params() -> tuple[np.array]:
    W1 = np.random.rand(10, 784) -0.5 # Creates a 10row * 784col 2d array where each value is a random val between -0.5 and 0.5
    b1 = np.random.rand(10, 1) -0.5# Bias vector
    W2 = np.random.rand(10, 10) -0.5# Creates a 10row * 784col 2d array where each value is a random val between -0.5 and 0.5
    b2 = np.random.rand(10, 1) -0.5# Bias vector
    return W1, b1, W2, b2

def activate(arr: np.array) -> np.array:
    return np.maximum(arr, 0)

def derActivate(arr: np.array) -> np.array:
    return arr > 0

def softmax(arr: np.array) -> np.array:
    return np.exp(arr) / sum(np.exp(arr))

def forwardProp(W1: np.array, b1: np.array, W2: np.array, b2: np.array, X: np.array) -> tuple[np.array]:
    Z1 = W1.dot(X) + b1
    A1 = activate(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def oneHot(labels: np.array) -> np.array:
    oneHotLabels = np.zeros((labels.size, labels.max()+1)) # labels.max() returns 9, +1 results in 10
    oneHotLabels[np.arange(labels.size), labels] = 1
    oneHotLabels = oneHotLabels.T
    return oneHotLabels

def backwardProp(Z1: np.array, A1: np.array, Z2: np.array, A2: np.array, W2: np.array, X: np.array,  Y: np.array) -> tuple[np.array]:
    oneHotLabels = oneHot(Y)
    dZ2 = A2 - oneHotLabels
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derActivate(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def updateParams(W1: np.array, b1: np.array, W2: np.array, b2: np.array, dW1: np.array, db1: np.array, dW2: np.array, db2: np.array, alpha: int) -> tuple[np.array]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def getPredictions(A2: np.array) -> np.array:
    return np.argmax(A2, 0)

def getAccuracy(predictions, labels: np.array) -> float:
    return np.sum(predictions == labels) / labels.size

def gradientDesc(inputs: np.array, labels: np.array, iterations: int, alpha: int):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, inputs)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W2, inputs, labels)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print(f'Iteration: {i}')
            print(f'Accuracy: {getAccuracy(getPredictions(A2), labels)}')
    return W1, b1, W2, b2

if __name__ == "__main__":
    W1, b1, W2, b2 = gradientDesc(trnInputs, trnLabels, 100, 0.1)