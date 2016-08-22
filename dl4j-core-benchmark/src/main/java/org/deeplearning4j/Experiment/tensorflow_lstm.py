#!/usr/bin/env python

"""
TensorFlow LSTM

 Reference:
    https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html

Data:
    Penn Tree Bank (https://www.cis.upenn.edu/~treebank/)
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

"""

import tensorflow as tf

class LSTM():

# TODO finish building
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


    def build_model(self):
        # TODO  multiple lstms...
        # lstm = rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        # stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers,
        #                                      state_is_tuple=False)
        #
        # initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
        # for i in range(num_steps):
        #     # The value of state is updated after processing each batch of words.
        #     output, state = stacked_lstm(words[:, i], state)


        lstm = rnn_cell.BasicLSTMCell(lstm_size)
        # Initial state of the LSTM memory.
        state = tf.zeros([batch_size, lstm.state_size])

        loss = 0.0
        for current_batch_of_words in words_in_dataset:
            # The value of state is updated after processing each batch of words.
            output, state = lstm(current_batch_of_words, state)

            # The LSTM output can be used to make next word predictions
            logits = tf.matmul(output, softmax_w) + softmax_b
            probabilities = tf.nn.softmax(logits)
            loss += loss_function(probabilities, target_words)

        # final_state = state
    def truncated_backprop(self):
        # Placeholder for the inputs in a given iteration.
        words = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm = rnn_cell.BasicLSTMCell(lstm_size)
        # Initial state of the LSTM memory.
        initial_state = state = tf.zeros([batch_size, lstm.state_size])

        for i in range(num_steps):
            # The value of state is updated after processing each batch of words.
            output, state = lstm(words[:, i], state)

            # The rest of the code.
            # ...

        final_state = state

    def dataset_iter(self):
        # A numpy array holding the state of LSTM after each batch of words.
        numpy_state = initial_state.eval()
        total_loss = 0.0
        for current_batch_of_words in words_in_dataset:
            numpy_state, current_loss = session.run([final_state, loss],
                                                    # Initialize the LSTM state from the previous iteration.
                                                    feed_dict={initial_state: numpy_state, words: current_batch_of_words})
            total_loss += current_loss