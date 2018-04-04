#!/usr/bin/env python3.5

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learningRate = 0.01
nb_entry = 28 * 28 # 28 pixels per 28 pixels
nb_hidden1 = 300
nb_hidden2 = 100
nb_out = 10 # 0, 1, 2, ..., 9
batch_size = 1000
n_epoch = 500


def main():
    mnist = input_data.read_data_sets("/tmp/data/")

    entry = tf.placeholder(tf.float32, shape=[None, nb_entry])
    layer1 = tf.layers.dense(entry, nb_hidden1, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, nb_hidden2, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, nb_out)

    expectedOuts = tf.placeholder(tf.int32, shape=[None])
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=expectedOuts)
    loss = tf.reduce_mean(xentropy)
    correct = tf.nn.in_top_k(out, expectedOuts, 1) # Return boolean matrix of correct predicted classes
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    gradientDescentOp = optimizer.minimize(loss)

    init = tf.global_variables_initializer() # Variable initilizer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables

        for i in range(n_epoch):
            if i % 10 == 0:
                test_accuracy = accuracy.eval(feed_dict={entry: mnist.test.images, \
                                                         expectedOuts: mnist.test.labels}) * 100
                print("Iteration", i, "\tTest Accuracy:", test_accuracy, "%")
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(gradientDescentOp, feed_dict={entry: X_batch, \
                                                   expectedOuts: y_batch})
            i += 1
        print("FINISHED", "%\tTest Accuracy:", test_accuracy, "%")

if __name__ == "__main__":
    main()
