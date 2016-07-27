#!/usr/bin/env python

"""
TensorFlow Lenet

 Reference:
    https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

TensorFlow install instructions: https://tensorflow.org/get_started/os_setup.html
MNIST tutorial: https://tensorflow.org/tutorials/mnist/tf/index.html
"""

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from six.moves import urllib, xrange
import os

NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
IMAGE_PIXELS = mnist.IMAGE_PIXELS
CORE_TYPE = 'GPU'
DTYPE = tf.float32
DEVICE = '/cpu:0' #if(CORE_TYPE == 'CPU') else '/gpu:0'
NUM_GPUS = 0 if(CORE_TYPE == 'CPU') else 1
CUDNN = True
DATA_DIR = os.getcwd() + "src/main/resources/tf_data/"
DATA_FORMAT = 'NHWC' # number examples, height, width, channels
ONE_HOT = True

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (11 * 60000)/66
tf.app.flags.DEFINE_integer('max_iter', 10000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('ffn1', 500, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 66, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_integer('test_batch_size', 66, 'Test batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('bias_learning_rate', 0.02, 'Initial bias rate.') #
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_float('decay_rate', 1e-3, 'Learning rate decay rate.')
tf.app.flags.DEFINE_float('policy_power', 0.75, 'Policy power.') # current inverse_time_decay is missing this as part of denom calc
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')

# TODO install latest github to access inverse | test code

# Tips Learned
# cpu BiasOp only support NHWC
# limits to using tf.float64 on certain functions - avoid
# cuDNN required for CNNs on GPU
# To maintain model in session prevents from setting different test batch size from training like other platform examples on GPU?

def load_data():
    return input_data.read_data_sets(DATA_DIR) if(ONE_HOT == False) else \
        input_data.read_data_sets(DATA_DIR, one_hot=True)


def _fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def _placeholder_inputs():
    """Generate placeholder variables to represent the input tensors.
    """
    images_placeholder = tf.placeholder(DTYPE, [None, IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(DTYPE, [None]) if(ONE_HOT == False) else \
        tf.placeholder(DTYPE, [None, NUM_CLASSES])
    return images_placeholder, labels_placeholder


def _init_weights(shape):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=FLAGS.seed, dtype=DTYPE), dtype=DTYPE)
    weight_decay = tf.mul(tf.nn.l2_loss(weights), FLAGS.l2, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return weights


def _inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    with tf.variable_scope('cnn1') as scope:
        images = tf.reshape(images, [FLAGS.batch_size, HEIGHT, WIDTH,  CHANNELS])
        depth1 = 20
        kernel = _init_weights([5, 5, CHANNELS, depth1])
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], "VALID", data_format= DATA_FORMAT,
                            use_cudnn_on_gpu=CUDNN) #VALID no padding
        biases = tf.Variable(tf.zeros([depth1], dtype=DTYPE), name='biases')
        conv1 = tf.nn.bias_add(conv, biases, name=scope.name, data_format=DATA_FORMAT)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=DATA_FORMAT, name='maxpool1')
    with tf.variable_scope('cnn2') as scope:
        depth2 = 50
        kernel = _init_weights([5, 5, depth1, depth2])
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], "VALID", data_format=DATA_FORMAT,
                            use_cudnn_on_gpu=CUDNN)
        biases = tf.Variable(tf.zeros([depth2], dtype=DTYPE), name='biases')
        conv2 = tf.nn.bias_add(conv, biases, name=scope.name, data_format=DATA_FORMAT)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=DATA_FORMAT, name='maxpool2')
    with tf.variable_scope('ffn1') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _init_weights([dim, FLAGS.ffn1])
        biases = tf.Variable(tf.zeros([FLAGS.ffn1], dtype=DTYPE), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _init_weights([FLAGS.ffn1, NUM_CLASSES])
        biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=DTYPE), name='biases')
        logits = tf.add(tf.matmul(hidden1, weights), biases, name=scope.name)
    return logits


def _score(logits, labels):
    """Calculates the loss from the logits and the labels.
    """
    labels = tf.to_int32(labels) if(ONE_HOT == False) else labels
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy') if(ONE_HOT == False) else \
        tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy') # works one hot
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def _trainer(loss):
    """Sets up the training Ops.
    """
    tf.scalar_summary(loss.op.name, loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Not exactly the same as in dl4j and caffe for learning policy inverse
    # learning_rate = tf.train.inverse_time_decay(FLAGS.learning_rate, global_step, 1, FLAGS.decay_rate, name="inverse_lr_policy")
    # optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training(train_data):
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = _placeholder_inputs()
        logits = _inference(images_placeholder)

        # Add to the Graph the Ops for loss calculation.
        loss = _score(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = _trainer(loss)

        sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': NUM_GPUS}))

        # Run the Op to initialize the variables.
        sess.run(tf.initialize_all_variables())

        # Start the training loop.
        train_time = time.time()
        for _ in xrange(FLAGS.max_iter):
            feed_dict = _fill_feed_dict(train_data, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        train_time = time.time() - train_time
        return sess, logits, images_placeholder, labels_placeholder, train_time


def _evaluation_straight(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    """
    labels = tf.to_int32(labels)
    correct = tf.nn.in_top_k(logits, labels, 10) # needs labels to be rank
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def _prediction(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_pred, tf.float32))


def do_eval(sess, logits, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    """
    correct_count = 0
    num_examples = data_set.num_examples
    if(ONE_HOT == False):
        correct_count = 0  # Counts the number of correct predictions.
        for _ in xrange(FLAGS.test_iter):
            feed_dict = _fill_feed_dict(data_set, images_placeholder, labels_placeholder)
            correct_count += sess.run(_evaluation_straight(logits, labels_placeholder), feed_dict=feed_dict)
        print("Accuracy 1: %0.04f" % (correct_count / num_examples))
    else:
        for _ in xrange(FLAGS.test_iter):
            feed_dict = _fill_feed_dict(data_set, images_placeholder, labels_placeholder)
            correct_count += sess.run(_prediction(logits, labels_placeholder), feed_dict=feed_dict)
        print("Accuracy: ", (correct_count / num_examples)) #(accuracy/FLAGS.test_iter))


def printTime(time_type, time):
    min = int(round(time/60))
    sec = int(round(time - min*60))
    milli = time * 1000
    print(time_type + ' load time: %s min %s sec | %s millisec' %(min, sec, milli))


def run():
    total_time = time.time()

    data_load_time = time.time()
    data_sets = load_data()
    data_load_time = time.time() - data_load_time

    sess, logits, images_placeholder, labels_placeholder, train_time = run_training(data_sets.train)

    test_time = time.time()
    # TODO eval is failing for gpu?
    do_eval(sess, logits, images_placeholder, labels_placeholder, data_sets.test)
    test_time = time.time() - test_time
    sess.close

    total_time = time.time() - total_time
    print("****************Example finished********************")
    printTime('Data load', data_load_time)
    printTime('Train', train_time)
    printTime('Test', test_time)
    printTime('Total', total_time)


if __name__ == "__main__":
    run()
