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


import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from six.moves import urllib, xrange
import os
import re
import numpy as np

import pdb

NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
IMAGE_PIXELS = mnist.IMAGE_PIXELS
CORE_TYPE = 'MULTI'
DTYPE = tf.float32
DEVICE = '/cpu:0'
NUM_GPUS = {'CPU': 0, 'GPU': 1, 'MULTI': 4}
CUDNN = True
DATA_DIR = os.getcwd() + "src/main/resources/tf_data/"
DATA_FORMAT = 'NHWC' # number examples, height, width, channels
ONE_HOT = True

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0

TOWER_NAME = 'lenet_tower'

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
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Decay factor.')
tf.app.flags.DEFINE_float('bias_learning_rate', 0.02, 'Initial bias rate.') #
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_float('decay_rate', 1e-3, 'Learning rate decay rate.')
tf.app.flags.DEFINE_float('policy_power', 0.75, 'Policy power.') # current inverse_time_decay is missing this as part of denom calc
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')

# TODO install latest github to access inverse | test code


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
    with tf.device(DEVICE):
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

        config = tf.ConfigProto(device_count={'GPU':NUM_GPUS[CORE_TYPE]})
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

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
    return tf.reduce_sum(tf.cast(correct_pred, tf.float32), 0)


def do_eval(sess, logits, images_placeholder, labels_placeholder, data):
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

'''
Multi-GPUs
'''
def tower_loss(data, scope, images_placeholder, labels_placeholder):
    """Calculate the total loss on a single tower running the CIFAR model.
    """
    # Build inference Graph.
    logits = _inference(images_placeholder)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = _score(logits, labels_placeholder)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    print losses
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss_name +' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            pdb.set_trace()
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def run_multi_training(data):
    """Train for a number of iterations."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        images_placeholder, labels_placeholder = _placeholder_inputs()

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.learning_rate_decay_factor, # tech initial learning rate higher than standard
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)
        #opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)

        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in xrange(NUM_GPUS[CORE_TYPE]):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = tower_loss(data, scope, images_placeholder, labels_placeholder)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # # Add a summary to track the learning rate.
        # summaries.append(tf.scalar_summary('learning_rate', lr))
        #
        # # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(
        #                 tf.histogram_summary(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        # saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        # summary_op = tf.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        train_time = time.time()
        for _ in xrange(FLAGS.max_iter):
            feed_dict = _fill_feed_dict(data, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # if step % 10 == 0:
            #     num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            #     examples_per_sec = num_examples_per_step / duration
            #     sec_per_batch = duration / FLAGS.num_gpus
            #
            #     format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
            #                   'sec/batch)')
            #     print (format_str % (datetime.now(), step, loss_value,
            #                          examples_per_sec, sec_per_batch))

            # if step % 100 == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)
            #
            # # Save the model checkpoint periodically.
            # if step % 1000 == 0 or (step + 1) == FLAGS.max_iter:
            #     checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=step)

    train_time = time.time() - train_time
    return sess, train_time, images_placeholder, labels_placeholder

def run():
    total_time = time.time()

    data_load_time = time.time()
    data_sets = load_data()
    data_load_time = time.time() - data_load_time

    if CORE_TYPE != 'MULTI':
        sess, logits, images_placeholder, labels_placeholder, train_time = run_training(data_sets.train)
    else:
        sess, train_time, images_placeholder, labels_placeholder = run_multi_training(data_sets.train)
        logits = _inference(images_placeholder)

    test_time = time.time()
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
