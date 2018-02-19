#!/usr/bin/env python3

import numpy as np
import matplotlib as plt

targets = np.matrix([42, 21])

def generateLinearTrainSet():
    x = np.random.rand(100, 1) # Generate random entries
    x = np.c_[np.ones((100, 1)), x] # Add 1 entry for each exemple (bias)
    y = x @ targets.T + np.random.randn(100, 1) # Compute exemples
    return x, y

def main():
    x, y = generateLinearTrainSet()
    theta = np.linalg.inv(x.T @ x) @ x.T @ y # Normal equation

    print("Expected:")
    print(targets)
    print("Got:")
    print(theta)

if __name__ == "__main__":
    main()
