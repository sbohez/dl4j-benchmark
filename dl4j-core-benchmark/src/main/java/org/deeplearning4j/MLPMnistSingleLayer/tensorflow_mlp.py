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
DEBUG = True

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
        weights = util.init_weights([IMAGE_PIXELS, FLAGS.hidden1_units], FLAGS.seed, FLAGS.l2)
        biases = tf.Variable(tf.zeros([FLAGS.hidden1_units], dtype=util.DTYPE), dtype=util.DTYPE, name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = util.init_weights([FLAGS.hidden1_units, NUM_CLASSES], FLAGS.seed, FLAGS.l2)
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
            images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)

            # Build model
            logits = _inference(images_placeholder)

            # Define loss and optimizer
            cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(logits)) # softmax & cross entropy
            train_op = util.setup_optimizer(cross_entropy, FLAGS.learning_rate, FLAGS.momentum)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            eval_correct = util.evaluation_topk(logits, labels_placeholder)

            # Train
            config = tf.ConfigProto(device_count={'GPU': num_gpus})
            sess = tf.InteractiveSession(config=config)

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            train_time = time.time()
            for iter in xrange(FLAGS.max_iter):
                start_time = time.time()
                feed_dict = util.fill_feed_dict(mnist.train, x, labels_placeholder, FLAGS.batch_size)
                _, loss_value = sess.run([train_op, cross_entropy], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if DEBUG:
                    if iter % 100 == 0:
                        # Print status to stdout.
                        print('Step %d: loss = %.2f (%.3f sec)' % (iter, loss_value, duration))
                        # Update the events file.
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, iter)
                        summary_writer.flush()

                        # Save a checkpoint and evaluate the model periodically.
                    if (iter + 1) % 1000 == 0 or (iter + 1) == FLAGS.max_steps:
                        # Evaluate against the training set.
                        print('Training Data Eval:')

                        util.do_eval(sess,
                                     eval_correct,
                                     images_placeholder,
                                     labels_placeholder,
                                     mnist.train)
                        # Evaluate against the test set.
                        print('Test Data Eval:')
                        util.do_eval(sess,
                                     eval_correct,
                                     images_placeholder,
                                     labels_placeholder,
                                     mnist.test)

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
    # TODO fix
    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers()
    # parser.add_argument('-core_type', action="Use CPU, GPU or MULTI for mulitple gpus"
    #
    # args = parser.parse_args()
    # run(args.core_type)
    run(sys.argv[1])
