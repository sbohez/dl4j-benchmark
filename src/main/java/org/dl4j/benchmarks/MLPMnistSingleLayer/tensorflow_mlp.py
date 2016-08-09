#!/usr/bin/env python

"""
TensorFlow MLP Mnist

Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
    See extensive documentation at http://tensorflow.org/tutorials/mnist/beginners/index.md

A very simple MNIST classifer.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from six.moves import urllib, xrange
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.benchmark_util as util


NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 1
ONE_HOT = True

TOWER_NAME = 'mlp_tower'

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (15 * 60000)/128
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run test.')
tf.app.flags.DEFINE_integer('hidden1_units', 1000, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 6e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')


def _inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    # Hidden 1
    with tf.variable_scope('hidden1'):
        weights = util._init_weights([IMAGE_PIXELS, FLAGS.hidden1_units], FLAGS.seed, FLAGS.l2)
        biases = tf.Variable(tf.zeros([FLAGS.hidden1_units], dtype=util.DTYPE), dtype=util.DTYPE, name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = util._init_weights([FLAGS.hidden1_units, NUM_CLASSES], FLAGS.seed, FLAGS.l2)
        biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=util.DTYPE), dtype=util.DTYPE, name='biases')
        logits = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
    return logits


def run(core_type):
    total_time = time.time()
    num_gpus = util.NUM_GPUS[core_type]

    data_load_time = time.time()
    # Import data
    mnist = util.load_data(input_data, ONE_HOT)
    data_load_time = time.time() - data_load_time
    with tf.Graph().as_default():
        x, y_ = util._placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)

        # Build model
        y = _inference(x)

        # Define loss and optimizer
        cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # softmax & cross entropy
        train_op = util._setup_optimizer(cross_entropy, FLAGS.learning_rate, FLAGS.momentum)

        # Train
        config = tf.ConfigProto(device_count={'GPU': num_gpus})
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.initialize_all_variables())

        train_time = time.time()
        for _ in xrange(FLAGS.max_iter):
            feed_dict = util._fill_feed_dict(mnist.train, x, y_, FLAGS.batch_size)
            _, loss_value = sess.run([train_op, cross_entropy], feed_dict=feed_dict)
        train_time = time.time() - train_time

    # Test trained model
    test_time = time.time()
    util.do_eval(sess, y, x, y_, mnist.test, ONE_HOT, FLAGS.test_iter, FLAGS.batch_size)
    test_time = time.time() - test_time
    sess.close


    total_time = time.time() - total_time
    print("****************Example finished********************")
    util.printTime('Data load', data_load_time)
    util.printTime('Train', train_time)
    util.printTime('Test', test_time)
    util.printTime('Total', total_time)


if __name__ == "__main__":
    # TODO fix
    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers()
    # parser.add_argument('-core_type', action="Use CPU, GPU or MULTI for mulitple gpus"
    #
    # args = parser.parse_args()
    # run(args.core_type)
    run(sys.argv[1])

