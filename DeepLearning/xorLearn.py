#!/usr/bin/env python3.5

import tensorflow as tf
import numpy as np

learningRate = 0.2
nb_entry = 2
nb_hidden1 = 2
nb_out = 1
nb_exemple = 4
threshold = 0.01

def get_dataset():
    X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [[0], [1], [1], [0]] # Reshape to have column values
    return X, y


def main():
    X, y = get_dataset()

    entry = tf.placeholder(tf.float32, shape=[None, nb_entry], name="entry")
    layer1 = tf.layers.dense(entry, nb_hidden1, name="hidden", activation=tf.nn.selu)
    out = tf.layers.dense(layer1, nb_out, name="output", activation=tf.nn.sigmoid)

    expectedOuts = tf.placeholder(tf.float32, shape=[None, nb_out], name="expectedOuts")
    loss = -tf.reduce_mean((expectedOuts * tf.log(out) + (1 - expectedOuts) * tf.log(1 - out)))

    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    gradientDescentOp = optimizer.minimize(loss)

    init = tf.global_variables_initializer() # Variable initilizer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables

        loss_value = loss.eval(feed_dict={entry: X, expectedOuts: y})
        print("Current loss:",(loss).eval(feed_dict={entry: X, expectedOuts: y}))
        i = 0
        while loss_value > threshold:
            if i % 100 == 0:
                print("Iteration", i)
                loss_value = loss.eval(feed_dict={entry: X, expectedOuts: y})
                print("Current loss:", loss_value)
            sess.run(gradientDescentOp, feed_dict={entry: X, expectedOuts: y})
            i += 1
        print(out.eval(feed_dict={entry: X}))

if __name__ == "__main__":
    main()
