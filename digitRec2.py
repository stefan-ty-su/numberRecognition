# 2 Hidden Layer Digit Recogniser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def getPredictions(A2: np.array) -> np.array:
    """
    Obtains the predicitons of a given set of observations

    Arguments:
    A2 -- Non-linear output of network

    Returns:
    np.array -- returns predicted outputs
    """
    return np.argmax(A2, 0)

def getAccuracy(predictions: np.array, labels: np.array) -> float:
    """
    Returns the accuracy of a given set of predictions compared to actual
    labels

    Arguments:
    predictions -- np.array of predictions
    labels -- np.array 

    Returns:
    float -- proportion of correct predictions
    """
    return np.sum(predictions == labels) / labels.size

class DigitRecogniser2:

    def __init__(self, filePath: str, layer1nodes: int, layer2nodes: int, iterations: int, alpha: int) -> None:
        """
        Digit Recogniser class, creates a neural network with 2 hidden layers

        Arguments:
        filepath -- path of the file for training data 
        layer1nodes -- determines the number of nodes in first hidden layer
        layer2nodes -- determines the number of nodes in the second hidden layer
        iterations -- number of training iterations
        alpha -- learning rate of the neural network

        Returns:
        None
        """
        self.iterations = iterations
        self.alpha = alpha

        self.data = np.array(pd.read_csv(filePath))
        self.m, self.n = self.data.shape # m is the number of observations, n is the number of features + 1 (label)
        np.random.shuffle(self.data)

        vldData = self.data[0:self.m//40].T
        self.vldLabels = vldData[0]
        self.vldInputs = vldData[1:self.n]/255.

        trnData = self.data[self.m//40:self.m].T
        self.trnLabels = trnData[0]
        self.trnInputs = trnData[1:self.n]/255.

        self.weights1 = np.random.rand(layer1nodes, 784) -0.5 # Creates a 10row * 784col 2d array where each value is a random val between -0.5 and 0.5
        self.biases1 = np.random.rand(layer1nodes, 1) -0.5 # Bias vector
        self.weights2 = np.random.rand(layer2nodes, layer1nodes) -0.5 # Creates a 10row * 10col 2d array where each value is a random val between -0.5 and 0.5
        self.biases2 = np.random.rand(layer2nodes, 1) -0.5 # Bias vector
        self.weights3 = np.random.rand(10, layer2nodes) -0.5
        self.biases3 = np.random.rand(10, 1) -0.5
    
    def relu(self, arr: np.array) -> np.array:
        """
        Non-linear activation function used in forward propagation.
        Current implementation uses Rectified Linear Unit (ReLU)

        Arguments:
        arr -- np.array representing linear outcome of from weights and biases

        Returns:
        np.array -- 2d np.array where each value is determined by function
        """
        return np.maximum(arr, 0)
    
    def dRelu(self, arr: np.array) -> np.array:
        """
        Derivative function of non-linear activation function (ReLU).

        Arguments:
        arr -- np.array representing linear outcome from back propagation

        Returns:
        np.array -- 2d np.array where each value is determined by function
        """
        return arr > 0
    
    def sigmoid(self, arr: np.array) -> np.array:
        """
        Non-linear activation function used in forward propagation.
        Current implementation uses Rectified Linear Unit (ReLU)

        Arguments:
        arr -- np.array representing linear outcome of from weights and biases

        Returns:
        np.array -- 2d np.array where each value is determined by function
        """
        return 1/(1+np.exp(-arr))
    
    def dSigmoid(self, arr: np.array) -> np.array:
        """
        Derivative function of non-linear activation function (ReLU).

        Arguments:
        arr -- np.array representing linear outcome from back propagation

        Returns:
        np.array -- 2d np.array where each value is determined by function
        """
        return self.sigmoid(arr) * (1 - self.sigmoid(arr))

    def softmax(self, arr: np.array) -> np.array:
        """
        softmax funciton used to determine value of output

        Arguments:
        arr -- np.array representing linear outcome of output nodes

        Returns:
        np.array -- np.array of results where the sum of results results in 1
        """
        return np.exp(arr) / sum(np.exp(arr))
    
    def forwardProp(self, inputs: np.array) -> tuple[np.array]:
        """
        Calculates linear and non-linear outcomesfor each layer and returns
        arrays for both.

        Arguments:
        inputs - np.array of input data

        Returns:
        tuple[np.array] - tuple of np.arrays in the order
            - linear outcome of layer 1
            - non-linear outcome of layer 1
            - linear outcome of layer 2
            - non-linear outcome of layer 2
            - linear outcome of layer 3 (out)
            - non-linear outcome of layer 3 (out)
        """
        linear1 = self.weights1.dot(inputs) + self.biases1
        nonLinear1 = self.relu(linear1)

        linear2 = self.weights2.dot(nonLinear1) + self.biases2
        nonLinear2 = self.relu(linear2)

        linearOut = self.weights3.dot(nonLinear2) + self.biases3
        nonLinearOut = self.softmax(linearOut)

        return linear1, nonLinear1, linear2, nonLinear2, linearOut, nonLinearOut
    
    def oneHotEncode(self, labels: np.array) -> np.array:
        """
        One hot encodes labels arr

        Arguments:
        labels -- labels of data

        Returns:
        np.array -- one hot encoded np.array
        """
        oneHotLabels = np.zeros((labels.size, labels.max()+1))
        oneHotLabels[np.arange(labels.size), labels] = 1
        oneHotLabels = oneHotLabels.T
        return oneHotLabels
    
    def backwardProp(self, Z1: np.array, A1: np.array, Z2: np.array, A2: np.array, A3: np.array, inputs: np.array, labels: np.array) -> tuple[np.array]:
        """
        Backwards propagation function, returns the loss gradients of parameters of layers

        Arguments:
        Z1 -- Linear outcome of first layer
        A1 -- Non-linear outcome of first layer
        Z2 -- Linear outcome of second layer
        A2 -- Non-linear outcome of second layer
        A3 -- Non-linear outcome of output layer
        inputs -- input data
        labels - labels of input data

        Returns:
        tuple[np.array] - tuple of np.arrays in the order:
            - loss gradient of layer 1 weights
            - loss gradient of layer 1 biases
            - loss gradient of layer 2 weights
            - loss gradient of layer 2 biases
        """
        oneHotLabels = self.oneHotEncode(labels)
        dZ3 = A3 - oneHotLabels
        dW3 = (1/self.m) * dZ3.dot(A2.T)
        db3 = (1/self.m) * np.sum(dZ3)

        dZ2 = self.weights3.T.dot(dZ3) * self.dRelu(Z2)
        dW2 = (1/self.m) * dZ2.dot(A1.T)
        db2 = (1/self.m) * np.sum(dZ2)

        dZ1 = self.weights2.T.dot(dZ2) * self.dRelu(Z1)
        dW1 = (1/self.m) * dZ1.dot(inputs.T)
        db1 = (1/self.m) * np.sum(dZ1)

        return dW1, db1, dW2, db2, dW3, db3
    
    def updateParams(self, dW1: np.array, db1: np.array, dW2: np.array, db2: np.array, dW3: np.array, db3: np.array):
        """
        Updates the networks weights and biases based on the back propagation
        loss gradients

        Arguemnts:
        """
        self.weights1 = self.weights1 - self.alpha * dW1
        self.biases1 = self.biases1 - self.alpha * db1
        self.weights2 = self.weights2 - self.alpha * dW2
        self.biases2 = self.biases2 - self.alpha * db2
        self.weights3 = self.weights3 - self.alpha * dW3
        self.biases3 = self.biases3 - self.alpha * db3
    
    def gradientDesc(self) -> None:
        """
        Gradient descent algorithm
        """
        for i in range(self.iterations):
            linear1, nlinear1, linear2, nlinear2, linear3, nlinear3 = self.forwardProp(self.trnInputs)
            dW1, db1, dW2, db2, dW3, db3 = self.backwardProp(linear1, nlinear1, linear2, nlinear2, nlinear3, self.trnInputs, self.trnLabels)
            self.updateParams(dW1, db1, dW2, db2, dW3, db3)
            if i % 50 == 0:
                print(f'Iteration: {i}')
                print(f'Accuracy: {getAccuracy(getPredictions(nlinear3), self.trnLabels)}')
            

if __name__ == "__main__":
    filePath = 'data/train.csv'
    network = DigitRecogniser2(filePath, 20, 10, 500, 0.1)
    network.gradientDesc()

    # Checking against validation data
    _, _, _, _, _, output = network.forwardProp(network.vldInputs)
    print(f'Accuracy on Validation Set: {getAccuracy(getPredictions(output), network.vldLabels)}')