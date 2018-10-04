from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from custom_ops import *
from tensorflow.python.platform import flags
import numpy as np

FLAGS = flags.FLAGS


class GridCell(object):
    def __init__(self):
        self.num_epochs = FLAGS.num_epochs
        self.batch_size = FLAGS.batch_size
        self.beta1 = FLAGS.beta1
        self.output_dir = FLAGS.output_dir
        self.place_dim, self.grid_cell_dim = FLAGS.place_dim, FLAGS.grid_cell_dim
        self.block_size = FLAGS.block_size

        self.lr = FLAGS.lr
        self.sigma = FLAGS.sigma
        self.log_step = FLAGS.log_step

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.model_dir = os.path.join(self.output_dir, 'model')

        self.velocity = FLAGS.velocity

    def build_model(self):
        self.place_before = tf.placeholder(shape=[None, self.place_dim], dtype=tf.float32)
        self.place_after = tf.placeholder(shape=[None, self.place_dim], dtype=tf.float32)

        weights = self.construct_weights()

        grid_cell_infer = tf.matmul(tf.transpose(weights['A']), tf.transpose(self.place_before))
        grid_cell_infer_after = tf.matmul(tf.diag(tf.ones(self.grid_cell_dim)) + weights['T'], grid_cell_infer)
        self.place_after_predicted = tf.matmul(weights['A'], grid_cell_infer_after)
        self.place_after_predicted = tf.transpose(self.place_after_predicted)

        regularizer_metabolic = tf.nn.l2_loss(grid_cell_infer_after)
        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.place_after - self.place_after_predicted), axis=0)) \
                    + 0.02 * tf.nn.l2_loss(weights['A']) + 0.00005 * regularizer_metabolic
        self.loss_mean, self.loss_update = tf.contrib.metrics.streaming_mean(self.loss)
        self.T_test = weights['T'][2:4, :2]

        # TODO: try different optimizers
        optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1)
        trainable_vars = tf.trainable_variables()

        grads_vars = optim.compute_gradients(self.loss, var_list=trainable_vars)
        self.grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in grads_vars]
        # update by mean of gradients
        self.apply_grads = optim.apply_gradients(grads_vars)

        tf.summary.scalar('loss', self.loss_mean)

        self.summary_op = tf.summary.merge_all()

    def construct_weights(self):
        weights = {}
        weights['A'] = tf.Variable(tf.truncated_normal([self.place_dim, self.grid_cell_dim], stddev=0.01), name='A')

        assert np.mod(self.grid_cell_dim, self.block_size) == 0
        num_block = int(self.grid_cell_dim / self.block_size)
        with tf.variable_scope('T', reuse=False):
            weights_T = []
            for i in range(num_block):
                weights_T.append(tf.Variable(tf.truncated_normal([self.block_size, self.block_size], stddev=0.01), name=str(i)))
                # weights_T.append(tf.Variable(tf.zeros([self.block_size, self.block_size]), name=str(i)))
            weights_T = block_diagonal(weights_T)
        # TODO: Check if block_diagonal function makes sense
        weights['T'] = weights_T
        return weights


