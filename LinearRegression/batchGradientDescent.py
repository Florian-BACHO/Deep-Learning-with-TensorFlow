#!/usr/bin/env python3.5

import numpy as np
import matplotlib.pyplot as plt

nbExemple = 100
nbIteration = 1000
learningRate = 0.1
targets = np.matrix([[42], [21]]) # Column vector of targeted weights

def generateLinearTrainSet():
    x = np.random.rand(nbExemple, 1) # Generate random entries
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std # Normalize
    x = np.c_[np.ones((nbExemple, 1)), x] # Add 1 entry for each exemple (bias)
    y = x @ targets + np.random.randn(nbExemple, 1) # Compute exemples
    return x, y, mean, std

def predict(theta, x):
    x_b = np.c_[np.ones((1, 1)), x] # Add 1 entry
    return x_b @ theta # Predict

def calculateGradient(theta, x, y):
    return 2. / nbExemple * x.T @ (x @ theta - y)

def executeBatchGradientDescent(x, y):
    theta = np.random.randn(2, 1) # Random initialization of weights with normal distribution

    for _ in range(nbIteration):
        gradients = calculateGradient(theta, x, y)
        theta = theta - learningRate * gradients
    return theta

def main():
    x, y, mean, std = generateLinearTrainSet()

    theta = executeBatchGradientDescent(x, y)

    print("Expected:")
    print(targets)
    print("Got:")
    print(theta)

    print("----------------")

    entries = np.matrix([-84])
    entries = (entries - mean) / std # Normalize
    print("Prediction with:")
    print(entries)
    print("Expected:")
    print(predict(targets, entries))
    print("Got:")
    print(predict(theta, entries))

    pred = x @ theta # Compute predictions on exemples
    xWithoutBias = np.delete(x, 0, 1) # Remove bias column to plot
    plt.plot(xWithoutBias, y, "b.")
    plt.plot(xWithoutBias, pred, "r-")
    plt.show()

if __name__ == "__main__":
    main()
