#!/usr/bin/env python3.5

import tensorflow as tf
import numpy as np

learningRate = 0.5
nb_entry = 2
nb_hidden1 = 2
nb_out = 2
nb_exemple = 4
threshold = 0.01

def get_dataset():
    X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [0, 1, 1, 0]
    return X, y


def main():
    X, y = get_dataset()

    entry = tf.placeholder(tf.float32, shape=[None, nb_entry], name="entry")
    layer1 = tf.layers.dense(entry, nb_hidden1, name="hidden", activation=tf.nn.selu)
    layer2 = tf.layers.dense(layer1, nb_out, name="hidden2")
    out = tf.nn.softmax(layer2)

    expectedOuts = tf.placeholder(tf.int32, shape=[None], name="expectedOuts")
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=expectedOuts)
    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    gradientDescentOp = optimizer.minimize(loss)

    init = tf.global_variables_initializer() # Variable initilizer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables

        loss_value = loss.eval(feed_dict={entry: X, expectedOuts: y})
        print("Current loss:", loss_value)
        i = 0
        while loss_value > threshold:
            if i % 1 == 0:
                print("Iteration", i)
                loss_value = loss.eval(feed_dict={entry: X, expectedOuts: y})
                print("Current loss:", loss_value)
            sess.run(gradientDescentOp, feed_dict={entry: X, expectedOuts: y})
            i += 1
        print(out.eval(feed_dict={entry: X}))

if __name__ == "__main__":
    main()
