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

class Lenet():

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
        self.one_hot = config['one_hot']
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
        ccn_depth1 = 20
        ccn_depth2 = 50
        ffn1_depth = 500

        with tf.variable_scope('cnn1') as scope:
            images = tf.reshape(self.images_placeholder, [self.batch_size, self.height, self.width, self.channelsss])
            kernel = self.init_weights([5, 5, self.channelss, ccn_depth1])
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], "VALID", data_format= self.data_format,
                                use_cudnn_on_gpu=self.use_cudnn) #VALID no padding
            biases = self.init_bias([ccn_depth1])
            conv1 = tf.identity(tf.nn.bias_add(conv, biases, data_format=self.data_format), name=scope.name)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               data_format=self.data_format, name='maxpool1')
        with tf.variable_scope('cnn2') as scope:
            kernel = self.init_weights([5, 5, ccn_depth1, ccn_depth2])
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], "VALID", data_format=self.data_format,
                                use_cudnn_on_gpu=self.use_cudnn)
            biases = self.init_bias([ccn_depth2])
            conv2 = tf.identity(tf.nn.bias_add(conv, biases, data_format=self.data_format), name=scope.name)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               data_format=self.data_format, name='maxpool2')
        with tf.variable_scope('ffn1') as scope:
            reshape = tf.reshape(pool2, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self.init_weights([dim, ffn1_depth])
            biases = self.init_bias([ffn1_depth])
            ffn1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self.init_weights([ffn1_depth, self.num_classes])
            biases = self.init_bias([self.num_classes])
            softmax_linear = tf.nn.softmax(tf.add(tf.matmul(ffn1, weights), biases, name=scope.name))
        self.model = softmax_linear


    def define_loss(self, labels):
        """Calculates the loss from the logits and the labels.
        """
        # TODO setup int16 for fp16 if needed
        labels = tf.to_int32(labels) if(self.one_hot is False and self.dtype == tf.float32) else labels
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.model, labels, name='xentropy') if(self.one_hot is False) else
                                       tf.nn.softmax_cross_entropy_with_logits(self.model, labels, name='xentropy'))
        tf.scalar_summary(cross_entropy.op.name, cross_entropy)
        self.loss = cross_entropy


    def setup_optimizer(self):
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)

