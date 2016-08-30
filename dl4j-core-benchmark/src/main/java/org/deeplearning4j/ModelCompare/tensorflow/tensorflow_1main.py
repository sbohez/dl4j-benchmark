'''
    TensorFlow Main Class

    Run models by passing in --model_type
    Current options are mlp and lenet
'''

import os
from six.moves import xrange
import logging
import re
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_lenet import Lenet
from tensorflow_mlp import MLP
from org.deeplearning4j.Utils.tensorflow_utils import print_time

DTYPE = tf.float32
DEVICE = '/cpu:0'
DATA_FORMAT = 'NHWC' # number examples, height, width, channels
TOWER_NAME = 'tower'
DATA_DIR = os.getcwd() + "/dl4j-core-benchmark/src/main/resources/tf_data/"
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

# create logger
LOGGER = logging.getLogger('simple_example')
LOGGER.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
LOGGER.addHandler(ch)

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (11 * 60000)/66
tf.app.flags.DEFINE_string('core_type', 'CPU', 'Directory to put the training data.')
tf.app.flags.DEFINE_string('model_type', 'mlp', 'Model to use either mlp or lenet.')
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')
tf.app.flags.DEFINE_boolean('one_hot', True, """Structure of target to be one hot vector or single scalar.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of GPUs.')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of GPUs.')
tf.app.flags.DEFINE_integer('height', 28, 'Height.')
tf.app.flags.DEFINE_integer('width', 28, 'Width.')
tf.app.flags.DEFINE_integer('channels', 1, 'Channels.')

model_config = {'lenet': {'batch_size': FLAGS.batch_size, 'num_classes': FLAGS.num_classes,
                          'height': FLAGS.height, 'width': FLAGS.width,'channels': FLAGS.channels,
                          'use_cudnn': FLAGS.core_type != 'CPU','learning_rate': 1e-2, 'momentum': 0.9,'l2': 1e-2, 'seed': FLAGS.seed,
                          'dtype': DTYPE, 'device': DEVICE, 'data_format': DATA_FORMAT, 'one_hot': FLAGS.one_hot},
                'mlp': {'batch_size': FLAGS.batch_size, 'num_classes': FLAGS.num_classes,
                        'height': FLAGS.height, 'width': FLAGS.width,'channels': FLAGS.channels,
                        'use_cudnn': False,'learning_rate': 6e-4, 'momentum': 0.9,'l2': 1e-4, 'seed': FLAGS.seed,
                     'dtype': DTYPE, 'device': DEVICE, 'data_format': DATA_FORMAT}}


############## Data ###############

def load_data(input_data, one_hot):
    if(one_hot is False):
        data = input_data.read_data_sets(DATA_DIR)
    else:
        data = input_data.read_data_sets(DATA_DIR, one_hot=True)
    return data


def placeholder_inputs(one_hot, num_pixels, num_classes):
    """Generate placeholder variables to represent the input tensors.
    """
    images_placeholder = tf.placeholder(DTYPE, [None, num_pixels])
    labels_placeholder = tf.placeholder(DTYPE, [None]) if(one_hot is False) else \
        tf.placeholder(DTYPE, [None, num_classes])
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
    """Fills the feed_dict for training the given step.
    """
    images_feed, labels_feed = data_set.next_batch(batch_size)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

############## Train ###############

def train(model, images_placeholder, labels_placeholder, data):
    if FLAGS.core_type != 'MULTI':
        sess, model, train_op = standard_train(model, labels_placeholder)
    else:
        sess, model, train_op = multi_train(model, labels_placeholder)
    train_time = time.time()
    LOGGER.debug("Train Model")
    for iter in xrange(FLAGS.max_iter):
        feed_dict = fill_feed_dict(data, images_placeholder, labels_placeholder, FLAGS.batch_size)
        _, loss_value = sess.run([train_op, model.loss], feed_dict=feed_dict)
        if iter % 100 == 0: LOGGER.debug('Iter %d: loss = %.2f' % (iter, loss_value))
    train_time = time.time() - train_time
    return sess, model, train_time


def standard_train(model, labels_placeholder):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model.define_loss(labels_placeholder)
    model.setup_optimizer()
    train_op = model.optimizer.minimize(model.loss, global_step)
    config = tf.ConfigProto(device_count={'GPU': FLAGS.num_gpus})
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.initialize_all_variables())
    return sess, model, train_op

############## Multi-GPU Functions ###############

def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    """
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)

        # Name each loss as '(raw)' and name the moving average version of the loss
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



def multi_train(model, labels_placeholder):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)
    model.setup_optimizer()
    opt = model.optimizer
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                model.define_loss(labels_placeholder)
                cross_entropy = model.loss
                tf.add_to_collection("losses", cross_entropy)
                tf.add_n(tf.get_collection('losses'), name='total_loss')
                loss = tower_loss(scope)
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()
                # Calculate the gradients for the batch of data on this tower.
                grads = opt.compute_gradients(loss)
                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

    # Calculate the mean of each gradient.
    grads = average_gradients(tower_grads)

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
                                                       log_device_placement=FLAGS.log_device_placement,
                                                       device_count={'GPU': FLAGS.num_gpus}))
    sess.run(init)
    return sess, model, train_op

############## Eval ###############

def evaluation_topk(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    """
    labels = tf.to_int32(labels)
    correct = tf.nn.in_top_k(logits, labels, 10) # needs labels to be rank
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def prediction(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_pred, DTYPE), 0)


def do_eval(sess, model, images_placeholder, labels_placeholder, data):
    """Runs one evaluation against the full epoch of data.
    """
    LOGGER.debug("Evaluate Model")
    correct_count = 0
    num_examples = data.num_examples
    for _ in xrange(FLAGS.test_iter):
        if FLAGS.one_hot is False:
            feed_dict = fill_feed_dict(data, images_placeholder, labels_placeholder)
            correct_count += sess.run(evaluation_topk(model, labels_placeholder), feed_dict=feed_dict)
        else:
            feed_dict = fill_feed_dict(data, images_placeholder, labels_placeholder, FLAGS.batch_size)
            correct_count += sess.run(prediction(model, labels_placeholder), feed_dict=feed_dict)

    print("Accuracy: %.2f" % ((correct_count / num_examples) * 100))


############## Run ###############

def run():
    total_time = time.time()

    FLAGS.num_gpus = 1 if FLAGS.core_type == 'GPU' else FLAGS.num_gpus

    LOGGER.debug("Load Data")
    data_load_time = time.time()
    data_sets = load_data(input_data, FLAGS.one_hot)
    data_load_time = time.time() - data_load_time


    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.one_hot, FLAGS.height*FLAGS.width*FLAGS.channels, FLAGS.num_classes)
        LOGGER.debug("Build Model")
        if FLAGS.model_type == 'lenet':
            model = Lenet(images_placeholder, model_config['lenet'])
            model.build_model()
        else:
            model = MLP(images_placeholder, model_config['mlp'])
            model.build_model()

        sess, model, train_time = train(model, images_placeholder, labels_placeholder, data_sets.train)
        test_time = time.time()
        do_eval(sess, model.model, images_placeholder, labels_placeholder, data_sets.test)
        test_time = time.time() - test_time

    sess.close

    total_time = time.time() - total_time
    print("****************Example finished********************")
    print_time('Data load', data_load_time)
    print_time('Train', train_time)
    print_time('Test', test_time)
    print_time('Total', total_time)


if __name__ == "__main__":
    run()