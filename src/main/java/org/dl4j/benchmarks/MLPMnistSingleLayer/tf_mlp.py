#!/usr/bin/env python

"""
TensorFlow MLP Mnist
Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

A very simple MNIST classifer.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
from six.moves import urllib, xrange

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 1
CORE_TYPE = 'GPU'
DTYPE = tf.float32

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (15 * 60000)/128
tf.app.flags.DEFINE_integer('max_iter', 7032, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('hidden1_units', 1000, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 6e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_int('seed', 42, 'Random seed.')


# TODO add gpu functionality & confirm accuracy | compare to other examples


def init_weights(shape):
    # Xavier weight initialization
    (fan_in, fan_out) = shape
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    weights = tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=DTYPE, seed=FLAGS.seed))
    weight_decay = tf.mul(tf.nn.l2_loss(weights), FLAGS.l2, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return weights


def inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights([IMAGE_PIXELS, FLAGS.hidden1_units])
        biases = tf.Variable(tf.zeros([FLAGS.hidden1_units]),  name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable([FLAGS.hidden1_units, NUM_CLASSES])
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden1, weights) + biases
    return logits


def run():
    start_time = time.time()
    # Hyper-parameters

    # Import data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x = tf.placeholder(DTYPE, [None, 784])
    y_ = tf.placeholder(DTYPE, [None, 10])

    # Build model
    logits = inference(x)

    # Define loss and optimizer
    cross_entropy = -tf.reduce_sum(y_*tf.log(logits)) # softmax & cross entropy
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = optimizer.minimize(cross_entropy, global_step=global_step)

    # Train
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    for _ in xrange(FLAGS.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_step.run({x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, DTYPE))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
    duration = time.time() - start_time
    print('Total train time: %s' % duration)
    sess.close

if __name__ == "__main__":
    run()

