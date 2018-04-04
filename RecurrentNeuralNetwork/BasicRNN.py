import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import sin

n_step = 20
n_input = 1
n_neurons = 100
n_outputs = 1
n_iteration = 500
learningRate = 0.01

n_train_ex = 100
n_test_ex = 10

def testFunction(x):
    return sin(x)

def getDataSet():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x = -5

    for i in range(n_train_ex): # Create an exemple
        tmp = []
        for j in range(n_step): # Append each time entry
            tmp.append([testFunction(x - n_step + j)])
        y_train.append([testFunction(x)])
        x_train.append(tmp)
        x += 0.1

    for i in range(n_test_ex):
        tmp = []
        for j in range(n_step):
            tmp.append([testFunction(x - n_step + j)])
        y_test.append([testFunction(x)])
        x_test.append(tmp)
        x += 0.1

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = getDataSet()
x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_outputs])

cell = tf.nn.rnn_cell.BasicRNNCell(n_neurons)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
logits = tf.layers.dense(states, n_outputs)

loss = tf.losses.mean_squared_error(labels=y, predictions=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()

    for iteration in range(n_iteration):
        sess.run(training_op, feed_dict={x: x_train, y: y_train})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={x: x_test, y: y_test})
            print(iteration, "\tMSE:", mse)

    plt.plot(y_train, "b.")
    plt.plot(logits.eval(feed_dict={x: x_train}), "r.")
    plt.plot(y_test, "b.")
    plt.plot(logits.eval(feed_dict={x: x_test}), "r.")
    plt.show()
