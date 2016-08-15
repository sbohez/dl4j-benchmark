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
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (15 * 60000)/128
tf.app.flags.DEFINE_string('core_type', 'CPU', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run test.')
tf.app.flags.DEFINE_integer('hidden1_units', 1000, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 6e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")


def _inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    util.LOGGER.debug("Build Model")
    with tf.variable_scope('hidden1'):
        weights = util.init_weights([IMAGE_PIXELS, FLAGS.hidden1_units], FLAGS.seed, FLAGS.l2)
        biases = util.init_bias([FLAGS.hidden1_units])
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.variable_scope('softmax_linear'):
        weights = util.init_weights([FLAGS.hidden1_units, NUM_CLASSES], FLAGS.seed, FLAGS.l2)
        biases = util.init_bias([NUM_CLASSES])
        logits = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
    return logits


def run():
    total_time = time.time()
    num_gpus = util.NUM_GPUS[FLAGS.core_type]

    util.LOGGER.debug("Load Data")
    data_load_time = time.time()
    mnist = util.load_data(input_data, ONE_HOT)
    data_load_time = time.time() - data_load_time
    with tf.Graph().as_default():
        util.LOGGER.debug("Load Data")
        images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)

        if FLAGS.core_type != 'MULTI':
            logits = _inference(images_placeholder)
            loss = -tf.reduce_sum(labels_placeholder*tf.log(logits)) # softmax & cross entropy
            train_op = util.setup_optimizer(loss, FLAGS.learning_rate, FLAGS.momentum)
            config = tf.ConfigProto(device_count={'GPU': num_gpus})
            sess = tf.InteractiveSession(config=config)
            sess.run(tf.initialize_all_variables())

        else:
            global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)
            opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
            tower_grads = []
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (util.TOWER_NAME, i)) as scope:
                        logits = _inference(images_placeholder)
                        cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(logits))
                        tf.add_to_collection("losses", cross_entropy)
                        tf.add_n(tf.get_collection('losses'), name='total_loss')

                        loss = util.tower_loss(scope)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

            # Calculate the mean of each gradient.
            grads = util.average_gradients(tower_grads)

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                                               log_device_placement=FLAGS.log_device_placement))
            sess.run(init)

        train_time = time.time()
        util.LOGGER.debug("Train Model")
        for iter in xrange(FLAGS.max_iter):
            feed_dict = util.fill_feed_dict(mnist.train, images_placeholder, labels_placeholder, FLAGS.batch_size)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            if iter % 100 == 0: util.LOGGER.debug('Iter %d: loss = %.2f' % (iter, loss_value))
        train_time = time.time() - train_time

        # Test trained model
        test_time = time.time()
        util.do_eval(sess, logits, images_placeholder, labels_placeholder, mnist.test, ONE_HOT, FLAGS.test_iter, FLAGS.batch_size)
        test_time = time.time() - test_time
        sess.close

        total_time = time.time() - total_time
        print("****************Example finished********************")
        util.printTime('Data load', data_load_time)
        util.printTime('Train', train_time)
        util.printTime('Test', test_time)
        util.printTime('Total', total_time)


if __name__ == "__main__":
    run()

