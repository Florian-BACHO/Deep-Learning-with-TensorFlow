#!/usr/bin/env python3.5

import numpy as np
import matplotlib.pyplot as plt

nbExemple = 100
targets = np.matrix([[42], [21]]) # Column vector of targeted weights

def generateLinearTrainSet():
    x = np.random.rand(nbExemple, 1) # Generate random entries
    x = np.c_[np.ones((nbExemple, 1)), x] # Add 1 entry for each exemple (bias)
    y = x @ targets + np.random.randn(nbExemple, 1) # Compute exemples
    return x, y

def predict(theta, x):
    x_b = np.c_[np.ones((1, 1)), x]
    return x_b @ theta

def main():
    x, y = generateLinearTrainSet()
    theta = np.linalg.inv(x.T @ x) @ x.T @ y # Normal equation

    print("Expected:")
    print(targets)
    print("Got:")
    print(theta)

    print("----------------")

    entries = np.matrix([-84])
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
