#!/usr/bin/env python

"""
TensorFlow MLP Mnist
Reference:

A very simple MNIST classifer.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

def init_weights(shape):
    (fan_in, fan_out) = shape
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def run():
    # Hyper-parameters
    learning_rate = 0.0006
    training_epochs = 15
    batch_size = 128
    momentum = 0.9
    l2=1e-4

    # Import data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Create the model
    hiddenW = init_weights(shape=[784, 1000])
    hiddenb = tf.Variable(tf.zeros([1, 1000]))
    W = init_weights(shape=[1000, 10])
    b = tf.Variable(tf.zeros([1, 10]))

    hidden1 = tf.nn.relu(tf.matmul(x, hiddenW) + hiddenb)
    y = tf.nn.softmax(tf.matmul(hidden1, W) + b)

    # Define loss and optimizer
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimizer.minimize(cross_entropy, global_step=global_step)

    # Train
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    time = datetime.now()
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_step.run({x: batch_xs, y_: batch_ys})
    time = datetime.now() - time

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
    print("Total train time: %s" % time)
    sess.close

if __name__ == "__main__":
    run()

