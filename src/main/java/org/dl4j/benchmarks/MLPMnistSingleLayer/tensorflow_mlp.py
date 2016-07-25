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
import os
import numpy as np
from six.moves import urllib, xrange

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 1
CORE_TYPE = 'GPU'
DTYPE = tf.float32

# '/gpu:1' if multiple
DEVICE = '/cpu:0' if(CORE_TYPE == 'CPU') else '/gpu:0'
NUM_GPUS = 0 if(CORE_TYPE == 'CPU') else 1
DATA_DIR = os.getcwd() + "src/main/resources/tf_data/"

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (15 * 60000)/128
tf.app.flags.DEFINE_integer('max_iter', 7032, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 78, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('hidden1_units', 1000, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 6e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')


def _init_weights(shape):
    # Xavier weight initialization
    (fan_in, fan_out) = shape
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    weights = tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, seed=FLAGS.seed, dtype=DTYPE))
    weight_decay = tf.mul(tf.nn.l2_loss(weights), FLAGS.l2, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return weights


def _inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = _init_weights([IMAGE_PIXELS, FLAGS.hidden1_units])
        biases = tf.Variable(tf.zeros([FLAGS.hidden1_units], dtype=DTYPE), dtype=DTYPE, name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = _init_weights([FLAGS.hidden1_units, NUM_CLASSES])
        biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=DTYPE), dtype=DTYPE, name='biases')
        logits = tf.matmul(hidden1, weights) + biases
    return logits

def _setup_train(y, y_):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # softmax & cross entropy
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(cross_entropy, global_step=global_step)

def _evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    """
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_pred, tf.float32))


def run():
    total_time = time.time()

    data_load_time = time.time()
    # Import data
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    data_load_time = time.time() - data_load_time

    x = tf.placeholder(DTYPE, [None, 784])
    y_ = tf.placeholder(DTYPE, [None, 10])

    # Build model
    y = _inference(x)

    # Define loss and optimizer
    train_step = _setup_train(y, y_)

    # Train
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) as sess:
    config = tf.ConfigProto(device_count={'GPU': NUM_GPUS})
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.initialize_all_variables())

    train_time = time.time()
    for _ in xrange(FLAGS.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_step.run({x: batch_xs, y_: batch_ys})
    train_time = time.time() - train_time

    # Test trained model
    test_time = time.time()
    correct_count = 0
    num_examples = mnist.test.num_examples
    # Define evaluation
    for _ in xrange(FLAGS.test_iter):
        batch_xs, batch_ys = mnist.test.next_batch(FLAGS.batch_size)
        correct_count += sess.run(_evaluation(y, y_), {x: batch_xs, y_: batch_ys})
    print("Accuracy: ", (correct_count / num_examples)) #(accuracy/FLAGS.test_iter))
    test_time = time.time() - test_time

    total_time = time.time() - total_time
    print("****************Example finished********************")
    print('Data load time: %s' % data_load_time*1000)
    print('Train time: %s' % train_time*1000)
    print('Test time: %s' % test_time*1000)
    print('Total time: %s' % total_time*1000)
    sess.close

if __name__ == "__main__":
    run()

