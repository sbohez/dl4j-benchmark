#!/usr/bin/env python

"""
TensorFlow MLP Mnist

Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
    See extensive documentation at http://tensorflow.org/tutorials/mnist/beginners/index.md

A very simple MNIST classifer.
"""
import tensorflow as tf

class MLP():

    def __init__(self, images_placeholder, config):
        self.batch_size = config['batch_size']
        self.num_classes = config['num_classes']
        self.height = config['height']
        self.width = config['width']
        self.channels = config['channels']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.l2 = config['l2']
        self.use_cudnn = config['use_cudnn']
        self.seed = config['seed']
        self.dtype = config['dtype']
        self.device = config['device']
        self.data_format = config['data_format']
        self.images_placeholder = images_placeholder

    def init_bias(self, shape):
        with tf.device(self.device):
            return tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0), dtype=self.dtype)

    def init_weights(self, shape):
        with tf.device(self.device):
            weights = tf.get_variable("weights", shape,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=self.seed, dtype=self.dtype), dtype=self.dtype)
        weight_decay = tf.mul(tf.nn.l2_loss(weights), self.l2, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return weights

    def build_model(self):
        """Build the MNIST model up to where it may be used for inference.
        """
        hidden1_units = 1000
        with tf.variable_scope('hidden1'):
            weights = self.init_weights([self.height*self.width*self.channels, hidden1_units])
            biases = self.init_bias([hidden1_units])
            hidden1 = tf.nn.relu(tf.matmul(self.images_placeholder, weights) + biases)
        with tf.variable_scope('softmax_linear'):
            weights = self.init_weights([hidden1_units, self.num_classes])
            biases = self.init_bias([self.num_classes])
            logits = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
        self.model = logits

    def define_loss(self, labels):
        self.loss = -tf.reduce_sum(labels*tf.log(self.model))

    def setup_optimizer(self):
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)


