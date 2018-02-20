#!/usr/bin/env python3.5

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

def get_dataset():
    housing = fetch_california_housing()
    nb_exemple, nb_entry = housing.data.shape

    data = np.c_[np.ones((nb_exemple, 1)), housing.data] # Add 1 entry
    X = tf.Variable(data)
    y = tf.Variable(housing.target.reshape(-1, 1)) # Reshape to have column values
    return X, y

def main():
    X, y = get_dataset()
    theta = tf.matrix_inverse(tf.transpose(X) @ X) @ tf.transpose(X) @ y # Create graph to apply normal equation

    init = tf.global_variables_initializer() # Variable initilizer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables
        theta_values = theta.eval() # Evaluate theta
    print(theta_values)

if __name__ == "__main__":
    main()
