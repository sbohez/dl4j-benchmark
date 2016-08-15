#!/usr/bin/env python

"""
TensorFlow Lenet

 Reference:
    https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

TensorFlow install instructions: https://tensorflow.org/get_started/os_setup.html
MNIST tutorial: https://tensorflow.org/tutorials/mnist/tf/index.html
"""

# Lessons Learned
# cpu BiasOp only support NHWC
# limits to using tf.float64 on certain functions - avoid
# cuDNN required for CNNs on GPU but hard to compile above 4
# To maintain model in session prevents from setting different test batch size from training like other platform examples on GPU?


import numpy as np
import re
import tensorflow as tf
import time
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.benchmark_util as util
import numpy as np


NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
IMAGE_PIXELS = mnist.IMAGE_PIXELS
ONE_HOT = True

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (11 * 60000)/66
tf.app.flags.DEFINE_string('core_type', 'CPU', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('ccn_depth1', 20, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('ccn_depth2', 50, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('ffn1', 500, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Decay factor.')
tf.app.flags.DEFINE_float('bias_learning_rate', 0.02, 'Initial bias rate.') #
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_float('decay_rate', 1e-3, 'Learning rate decay rate.')
tf.app.flags.DEFINE_float('policy_power', 0.75, 'Policy power.') # current inverse_time_decay is missing this as part of denom calc
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train', """Directory where to read model checkpoints.""")


def _inference(images, use_cudnn):
    """Build the MNIST model up to where it may be used for inference.
    """
    util.LOGGER.debug("Build Model")
    with tf.variable_scope('cnn1') as scope:
        images = tf.reshape(images, [FLAGS.batch_size, HEIGHT, WIDTH,  CHANNELS])
        kernel = util.init_weights([5, 5, CHANNELS, FLAGS.ccn_depth1], FLAGS.seed, FLAGS.batch_size)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], "VALID", data_format= util.DATA_FORMAT,
                            use_cudnn_on_gpu=use_cudnn) #VALID no padding
        biases = util.init_bias([FLAGS.ccn_depth1])
        # bias = tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT)
        conv1 = tf.identity(tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT), name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=util.DATA_FORMAT, name='maxpool1')
    with tf.variable_scope('cnn2') as scope:
        kernel = util.init_weights([5, 5, FLAGS.ccn_depth1, FLAGS.ccn_depth2], FLAGS.seed, FLAGS.batch_size)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], "VALID", data_format=util.DATA_FORMAT,
                            use_cudnn_on_gpu=use_cudnn)
        biases = util.init_bias([FLAGS.ccn_depth2])
        # bias = tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT)
        conv2 = tf.identity(tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT), name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=util.DATA_FORMAT, name='maxpool2')
    with tf.variable_scope('ffn1') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = util.init_weights([dim, FLAGS.ffn1], FLAGS.seed, FLAGS.batch_size)
        biases = util.init_bias([FLAGS.ffn1])
        ffn1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = util.init_weights([FLAGS.ffn1, NUM_CLASSES], FLAGS.seed, FLAGS.batch_size)
        biases = util.init_bias([NUM_CLASSES])
        softmax_linear = tf.nn.softmax(tf.add(tf.matmul(ffn1, weights), biases, name=scope.name))
    return softmax_linear


def _setup_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    """
    # TODO setup int16 for fp16 if needed
    labels = tf.to_int32(labels) if(ONE_HOT is False and util.DTYPE == tf.float32) else labels
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy') if(ONE_HOT is False) else
                                   tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy'))

    if FLAGS.core_type != "MULTI":
        tf.scalar_summary(cross_entropy.op.name, cross_entropy)
    else:
        tf.add_to_collection("losses", cross_entropy)
        tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy


def train(train_data, num_gpus, use_cudnn, images_placeholder, labels_placeholder):
    """Train for a number of iterations."""
    logits = _inference(images_placeholder, use_cudnn)

    loss = _setup_loss(logits, labels_placeholder)
    train_op = util.setup_optimizer(loss, FLAGS.learning_rate, FLAGS.momentum)

    config = tf.ConfigProto(device_count={'GPU': num_gpus})
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.initialize_all_variables())

    # Start the training loop.
    train_time = time.time()
    util.LOGGER.debug("Train Model")
    for iter in xrange(FLAGS.max_iter):
        feed_dict = util.fill_feed_dict(train_data, images_placeholder, labels_placeholder, FLAGS.batch_size)
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        # Write the summaries and print an overview fairly often.
        if iter % 100 == 0: util.LOGGER.debug('Iter %d: loss = %.2f (%.3f sec)' % (iter, loss_value, 0.0))
    train_time = time.time() - train_time
    return sess, logits, train_time


def multi_train(data, num_gpus, use_cudnn, images_placeholder, labels_placeholder):
    """Train for a number of iterations."""
    # Create a variable to count the number of train() calls.
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    # num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    #
    # # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(FLAGS.learning_rate, # tech initial learning rate higher than standard
    #                                 global_step,
    #                                 decay_steps,
    #                                 FLAGS.learning_rate_decay_factor,
    #                                 staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    # opt = tf.train.GradientDescentOptimizer(lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (util.TOWER_NAME, i)) as scope:
                logits = _inference(images_placeholder, use_cudnn)
                _ = _setup_loss(logits, labels_placeholder)
                # Calculate the loss for one tower. One model constructed per tower and variables shared across
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

    # tf.train.start_queue_runners(sess=sess) TODO add number threads to loading data to load on mult cpus

    train_time = time.time()
    util.LOGGER.debug("Train Model")
    for iter in xrange(FLAGS.max_iter):
        feed_dict = util.fill_feed_dict(data, images_placeholder, labels_placeholder, FLAGS.batch_size)
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        if iter % 100 == 0: util.LOGGER.debug('Iter %d: loss = %.2f' % (iter, loss_value))
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    train_time = time.time() - train_time
    return sess, logits, train_time


def run():
    total_time = time.time()

    util.LOGGER.debug("Load Data")
    data_load_time = time.time()
    data_sets = util.load_data(input_data, ONE_HOT)
    data_load_time = time.time() - data_load_time

    num_gpus = util.NUM_GPUS[FLAGS.core_type]
    use_cudnn = True if (FLAGS.core_type != "CPU") else False

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)
        if FLAGS.core_type != 'MULTI':
            sess, logits, train_time = train(data_sets.train, num_gpus, use_cudnn, images_placeholder, labels_placeholder)
        else:
            sess, logits, train_time = multi_train(data_sets.train, num_gpus, use_cudnn, images_placeholder, labels_placeholder)

        test_time = time.time()
        util.do_eval(sess, logits, images_placeholder, labels_placeholder, data_sets.test, ONE_HOT, FLAGS.test_iter, FLAGS.batch_size)
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
