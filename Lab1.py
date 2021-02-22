#!/Users/monal/CS373/bin/python3

# Delete the shebang for your use, I had a virtual environment set up for my CS373 class so I'm using that.

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn import datasets
import pandas

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"

# ALGORITHM = "custom_net"
#
#
ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1, layers=2, actFunction="sig"):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.layers = layers
        self.actFunction = actFunction

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return (self.__sigmoid(x)) * (1 - self.__sigmoid(x))

    def __relu(self, x):
        return np.maximum(x, 0)

    def __reluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):
        batchX = self.__batchGenerator(xVals, mbs)
        batchY = self.__batchGenerator(yVals, mbs)
        for currEpoch in range(epochs):
            for currBatchX, currBatchY in zip(batchX, batchY):
                L1out, L2out = self.__forward(currBatchX)
                L2e = L2out - currBatchY
                if self.actFunction == "sig":
                    L2d = L2e * self.__sigmoidDerivative(L2out)
                else:
                    L2d = L2e * self.__reluDerivative(L2out)
                L1e = np.dot(L2d, np.transpose(self.W2))
                if self.actFunction == "sig":
                    L1d = L1e * self.__sigmoidDerivative(L1out)
                else:
                    L1d = L1e * self.__reluDerivative(L1out)
                L1a = (np.dot(np.transpose(currBatchX), L1d)) * self.lr
                L2a = (np.dot(np.transpose(L1out), L2d)) * self.lr
                self.W1 -= L1a
                self.W2 -= L2a

    # Forward pass.
    def __forward(self, inp):
        if self.actFunction == "sig":
            layer1 = self.__sigmoid(np.dot(inp, self.W1))
            layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        else:
            layer1 = self.__relu(np.dot(inp, self.W1))
            layer2 = self.__relu(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        # print("Works")
        # print(xTest)
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    # np.set_printoptions(threshold=np.inf)
    # print(xTrain)
    return (xTrain, yTrain), (xTest, yTest)


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw  # TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    # np.set_printoptions(threshold=np.inf)
    xTrain = xTrain / 255
    xTest = xTest / 255
    xTrain = xTrain.reshape(60000, IMAGE_SIZE)
    xTest = xTest.reshape(10000, IMAGE_SIZE)
    # print(xTrain)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return (xTrain, yTrainP), (xTest, yTestP)


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")

        inputSize = IMAGE_SIZE
        outputSize = NUM_CLASSES
        neuronsPerLayer = 784
        learningRate = 0.1
        activationFunction = "sig"
        totalLayers = 2

        customNet = NeuralNetwork_2Layer(inputSize, outputSize, neuronsPerLayer, learningRate, totalLayers,
                                         activationFunction)

        customNet.train(xTrain, yTrain)
        return customNet

    elif ALGORITHM == "tf_net":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        model.fit(xTrain, yTrain, epochs=75)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        # print(data)
        # data.set_printoptions(threshold=data.inf)
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds, iris):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    confusionMatrix = np.zeros([preds.shape[1], preds.shape[1]])
    acc = 0
    for i in range(preds.shape[0]):
        predictedValue = np.argmax(preds[i])
        actualValue = np.argmax(yTest[i])
        confusionMatrix[actualValue][predictedValue] += 1
        if predictedValue == actualValue:
            acc = acc + 1

    iteration = 0
    if iris is False:
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        iteration = 10
    else:
        labels = ["0", "1", "2"]
        iteration = 3
    x = confusionMatrix.astype(int)

    df = pandas.DataFrame(x, columns=labels, index=labels)

    f1Score = []
    predictedRow = df.sum(axis=0)
    actualRow = df.sum(axis=1)

    for i in range(iteration):
        if predictedRow[i] == 0:
            f1Score.append(0)
            continue
        truePositive = x[i][i]
        falsePositive = predictedRow[i] - truePositive
        falseNegative = actualRow[i] - truePositive
        precision = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)
        f1 = (2 * precision * recall) / (precision + recall)
        f1Score.append(f1)

    df.loc['Total', :] = df.sum(axis=0)
    df.loc[:, 'Total'] = df.sum(axis=1)
    pandas.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    print()
    newdf = pandas.DataFrame(f1Score, columns=["F1Score"], index=labels)
    print(newdf)
    print()

    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds, False)

    # In order to run this with IRIS dataset, uncomment the lines below.

    # irisData = datasets.load_iris()
    # X = irisData.data
    # Y = irisData.target
    # xTrain, xTest, yTrain, yTest = train_test_split(X, Y)
    # yTrain = to_categorical(yTrain, 3)
    # yTest = to_categorical(yTest, 3)
    # nn = NeuralNetwork_2Layer(4, 3, 150, learningRate=0.01)
    # nn.train(xTrain, yTrain, epochs=100, mbs=15)
    # preds = nn.predict(xTest)
    # evalResults((xTest, yTest), preds, True)


if __name__ == '__main__':
    main()
