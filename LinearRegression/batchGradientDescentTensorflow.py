#!/usr/bin/env python3.5

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

nbIteration = 1000
learningRate = 0.01

def get_dataset():
    housing = fetch_california_housing()
    nb_exemple, nb_entry = housing.data.shape

    data = housing.data # Add 1 entry
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std
    data = np.c_[np.ones((nb_exemple, 1)), data] # Add 1 entry
    X = tf.Variable(data, dtype=tf.float32)
    y = tf.Variable(housing.target.reshape(-1, 1), dtype=tf.float32) # Reshape to have column values
    return X, y, nb_exemple, nb_entry

def main():
    X, y, nb_exemple, nb_entry = get_dataset()
    theta = tf.Variable(tf.random_uniform([nb_entry + 1, 1], -1., 1.), \
                        validate_shape=False) # Initialize theta randomly

    pred = X @ theta
    error = pred - y
    mse = tf.reduce_mean(tf.square(error))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    gradientDescentOp = optimizer.minimize(mse)

    init = tf.global_variables_initializer() # Variable initilizer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables

        for i in range(nbIteration):
            if i % 100 == 0:
                print("Iteration", i)
                print("Current loss:", mse.eval())
            sess.run(gradientDescentOp)
        best_theta = theta.eval()
        print(best_theta)

if __name__ == "__main__":
    main()
