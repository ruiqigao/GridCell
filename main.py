from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
from gridcell import GridCell
from gridcell_multidir import GridCell_multidir
from gridcell_multidir_3d import GridCell_multidir_3d
from data_io import Data_Generator
from utils import *
from path_planning import Path_planning, perform_path_planning
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import argparse
from scipy.io import savemat
from mayavi.mlab import *


# tf.set_random_seed(1234)
# np.random.seed(0)

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--batch_size', type=int, default=200000, help='Batch size of training images')
parser.add_argument('--num_epochs', type=int, default=8000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate for descriptor')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')

# simulated data parameters
parser.add_argument('--place_size', type=float, default=1.0, help='Size of the square place')
parser.add_argument('--max_vel1', type=float, default=39, help='maximum of velocity in loss1')
parser.add_argument('--min_vel1', type=float, default=1, help='minimum of velocity in loss1')
parser.add_argument('--max_vel2', type=float, default=3, help='maximum  of velocity in loss2')
parser.add_argument('--min_vel2', type=float, default=1, help='minimum of velocity in loss2')
parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.1], help='sd of gaussian kernel')
parser.add_argument('--num_data', type=int, default=200000, help='Number of simulated data points') # 30000
parser.add_argument('--dtype1', type=int, default=1, help='type of loss1')

# model parameters
parser.add_argument('--place_dim', type=int, default=64000, help='Dimensions of place, should be N^3')
parser.add_argument('--num_group', type=int, default=10, help='Number of groups of grid cells')  # 16
parser.add_argument('--block_size', type=int, default=6, help='Size of each block')
parser.add_argument('--iter', type=int, default=0, help='Number of iter')
parser.add_argument('--lamda', type=float, default=0.1, help='Hyper parameter to balance two loss terms')  # 0.1
parser.add_argument('--GandE', type=float, default=1, help='1: Gaussian kernel; 0: Exponential kernel')
parser.add_argument('--lamda2', type=float, default=5000, help='Hyper parameter to balance two loss terms')
parser.add_argument('--motion_type', type=str, default='continuous', help='True if in testing mode')
parser.add_argument('--num_step', type=int, default=1, help='Number of steps in path integral')
parser.add_argument('--save_memory', type=bool, default=False, help='True if in testing mode')

# utils train
parser.add_argument('--training_output_dir', type=str, default='training_result', help='The output directory for saving training results')
parser.add_argument('--testing_output_dir', type=str, default='testing_result', help='The output directory for saving testing results')
parser.add_argument('--log_step', type=int, default=200, help='Number of mini batches to save output results')  # 500

# utils test
parser.add_argument('--mode', type=str, default='0', help='0: training / 1: visualizing /  2: path integral')
parser.add_argument('--test_num', type=int, default=30, help='Number of testing steps used in path integral')
parser.add_argument('--project_to_point', type=bool, default=False, help='True if in testing path integral mode')
parser.add_argument('--ckpt', type=str, default='model.ckpt-7999', help='Checkpoint path to load')
parser.add_argument('--num_testing_path_integral', type=int, default=1000, help='Number of testing cases for path integral')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def train(model, sess, output_dir):

    log_dir = os.path.join(output_dir, 'log')

    if FLAGS.GandE == 1:
        model_dir = os.path.join(output_dir, 'gau_model')
    elif FLAGS.GandE == 0:
        model_dir = os.path.join(output_dir, 'exp_model')

    syn_dir = os.path.join(output_dir, 'learned_patterns')
    syn_path_dir = os.path.join(output_dir, 'path_integral')

    log_file = os.path.join(output_dir, 'testing_error.txt')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    if not tf.gfile.Exists(syn_dir):
        tf.gfile.MakeDirs(syn_dir)

    if not tf.gfile.Exists(syn_path_dir):
        tf.gfile.MakeDirs(syn_path_dir)


    # build model
    model.build_model()
    model.path_integral(FLAGS.test_num)
    # planning_model = Path_planning(model)

    num_batch = int(np.ceil(FLAGS.num_data / FLAGS.batch_size))

    lamda_list = np.linspace(FLAGS.lamda, FLAGS.lamda, FLAGS.num_epochs)

    # initialize training
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=20)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # make graph immutable
    tf.get_default_graph().finalize()

    # store graph in protobuf
    with open(model_dir + '/graph.proto', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))

    data_generator = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval,
                                    to_use_3D_map=True)

    place_pair_val1 = data_generator.generate(1000, dtype=FLAGS.dtype1)
    place_seq_val2 = data_generator.generate(1000, velocity=model.velocity2, num_step=model.num_step, dtype=2)
    # train
    start_time = time.time()
    for epoch in range(FLAGS.num_epochs):
        if epoch < FLAGS.iter:
            lamda_list[epoch] = 0
        place_pair1 = data_generator.generate(FLAGS.num_data, dtype=FLAGS.dtype1)
        place_seq2 = data_generator.generate(FLAGS.num_data, velocity=model.velocity2, num_step=model.num_step, dtype=2)

        loss1_avg, loss2_avg, reg_avg, loss3_avg, loss4_avg = [], [], [], [], []
        for minibatch in range(num_batch):
            start_idx = minibatch * FLAGS.batch_size
            end_idx = (minibatch + 1) * FLAGS.batch_size

            # update weights
            feed_dict = dict()
            feed_dict.update({model.place_before1: place_pair1['before'][start_idx: end_idx],
                              model.place_after1: place_pair1['after'][start_idx: end_idx],
                              model.vel1: place_pair1['vel'][start_idx: end_idx],
                              model.place_seq2: place_seq2['seq'][start_idx: end_idx],
                              model.lamda: lamda_list[epoch]})

            feed_dict[model.vel2] = place_seq2['vel'][start_idx: end_idx] if model.motion_type == 'continuous' \
                else place_seq2['vel_idx'][start_idx: end_idx]

            summary, loss1, loss2, reg, loss3, loss4, dp1, dp2 = sess.run([model.summary_op, model.loss1,
                                                                           model.loss2, model.reg, model.loss3,
                                                                           model.loss4,
                                                                           model.dp1, model.dp2, model.loss_update,
                                                                           model.apply_grads], feed_dict=feed_dict)[:8]

            # regularize weights
            #if epoch > 4000:
            #    sess.run(model.norm_grads)

            loss1_avg.append(loss1)
            loss2_avg.append(loss2)
            reg_avg.append(reg)
            loss3_avg.append(loss3)
            loss4_avg.append(loss4)

        writer.add_summary(summary, epoch)
        writer.flush()
        if epoch % 10 == 0:
            loss1_avg, loss2_avg, loss3_avg, loss4_avg, reg_avg = np.mean(np.asarray(loss1_avg)), np.mean(
                np.asarray(loss2_avg)), \
                                                                  np.mean(np.asarray(loss3_avg)), np.mean(
                np.asarray(loss4_avg)), \
                                                                  np.mean(np.asarray(reg_avg))
            feed_dict = dict()
            feed_dict.update({model.place_before1: place_pair_val1['before'],
                              model.place_after1: place_pair_val1['after'],
                              model.vel1: place_pair_val1['vel'],
                              model.place_seq2: place_seq_val2['seq'],
                              model.lamda: lamda_list[epoch]})
            feed_dict[model.vel2] = place_seq_val2['vel'][start_idx: end_idx] if model.motion_type == 'continuous' \
                else place_seq_val2['vel_idx'][start_idx: end_idx]
            loss_val = sess.run(model.loss, feed_dict=feed_dict)
            end_time = time.time()
            print(
                '#{:s} Epoch #{:d}, train loss1: {:.4f}, train loss2: {:.4f}, reg: {:.4f}, val loss: {:.4f} time: {:.2f}s'
                .format(output_dir, epoch, loss1_avg, loss2_avg, reg_avg, loss_val, end_time - start_time))

            start_time = time.time()

        # report a testing error in the task of path integral and record it in a file
        if epoch + 1 == FLAGS.num_epochs or (epoch + 1) % FLAGS.log_step == 0:

            print("****************** saving check point and computing testing error in path integral ****************")

            # save check point
            saver.save(sess, "%s/%s" % (model_dir, 'model.ckpt'), global_step=epoch)

            # store learned patterns
            visualize_3D_grid_cell(model, sess, syn_dir, epoch)

            # show one case of testing
            place_seq_test_single = data_generator.generate(1, velocity=model.velocity2, num_step=FLAGS.test_num,
                                                            dtype=2, test=True)
            test_path_integral(model, sess, place_seq_test_single, visualize=True, test_dir=syn_path_dir, epoch=epoch)

            # compute a testing error on a number of testing cases
            place_seq_test = data_generator.generate(FLAGS.num_testing_path_integral, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2,
                                                     test=True)
            err = test_path_integral(model, sess, place_seq_test)

            print("****************** (epoch %s) error of path integral in %s testing cases: %s" % (str(epoch), str(FLAGS.num_testing_path_integral), str(err)))

            if log_file is not None:
                with open(log_file, "a") as f:
                    print('epoch = %d , error = %02f' % (epoch, err), file=f)


def test_path_integral(model, sess, place_seq_test, visualize=False, test_dir=None, epoch=None):

    test_num = place_seq_test['seq'].shape[1] - 1
    err = np.zeros(shape=len(place_seq_test['seq']))

    place_min, place_max = 100, -100

    for i in range(len(place_seq_test['seq'])):
        feed_dict = {model.place_init_test: place_seq_test['seq'][i, 0],
                     model.vel2_test: place_seq_test['vel'][i]} if model.motion_type == 'continuous' \
            else {model.place_init_test: place_seq_test['seq'][i, 0],
                  model.vel2_test: place_seq_test['vel_idx'][i]}
        place_seq_pd_pt_value, place_seq_predict_value, place_seq_predict_gp_value = \
            sess.run([model.place_seq_pd_pt, model.place_seq_pd, model.place_seq_pd_gp], feed_dict=feed_dict)

        place_seq_gt = place_seq_test['seq'][i, 1:]
        err[i] = np.mean(np.sqrt(np.sum((place_seq_gt - place_seq_pd_pt_value) ** 2, axis=1)))

        if place_seq_predict_value.min() < place_min:
            place_min = place_seq_predict_value.min()
        if place_seq_predict_value.max() > place_max:
            place_max = place_seq_predict_value.max()

        if visualize:

            if not tf.gfile.Exists(test_dir):
               tf.gfile.MakeDirs(test_dir)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(place_seq_gt[:, 0], place_seq_gt[:, 1], place_seq_gt[:, 2], color="blue", label='ground truth')
            ax.scatter(place_seq_gt[0, 0], place_seq_gt[0, 1], place_seq_gt[0, 2], color="blue", marker='o')
            ax.scatter(place_seq_gt[-1, 0], place_seq_gt[-1, 1], place_seq_gt[-1, 2], color="blue", marker='x')
            ax.plot(place_seq_pd_pt_value[:, 0], place_seq_pd_pt_value[:, 1], place_seq_pd_pt_value[:, 2],
                       linestyle='dashed', color="red", label='predicted')
            ax.scatter(place_seq_pd_pt_value[0, 0], place_seq_pd_pt_value[0, 1], place_seq_pd_pt_value[0, 2], color="red",
                       marker='o')
            ax.scatter(place_seq_pd_pt_value[-1, 0], place_seq_pd_pt_value[-1, 1], place_seq_pd_pt_value[-1, 2], color="red",
                       marker='x')
            ax.legend()
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            #ax.grid(False)
            plt.savefig(os.path.join(test_dir, str(epoch) + '_id_' + str(i) + '.png'))
            plt.close()

    #print("place max: " + str(place_max) + "place min: " + str(place_min))
    err = np.mean(err)

    return err


def visualize_3D_grid_cell(model, sess, test_dir, epoch=0, slice_to_show=20):

    # only showing one 2D slice of the 3D grid patterns
    weights_A_value = sess.run(model.weights_A)
    if not tf.gfile.Exists(test_dir):
        tf.gfile.MakeDirs(test_dir)
    np.save(os.path.join(test_dir, 'weights.npy'), weights_A_value)

    # print out A
    weights_A_value_transform = weights_A_value.transpose(3, 0, 1, 2)
    # fig_sz = np.ceil(np.sqrt(len(weights_A_value_transform)))


    plt.figure(figsize=(model.block_size, model.num_group))
    for i in range(len(weights_A_value_transform)):
        weight_to_draw = weights_A_value_transform[i]
        plt.subplot(model.num_group, model.block_size, i + 1)

        # showing one slice (2D) of 3D grid patterns
        weight_to_draw_all = weight_to_draw[slice_to_show, :, :]
        draw_heatmap_2D(weight_to_draw_all, vmin=weight_to_draw_all.min(), vmax=weight_to_draw_all.max())
        
    plt.savefig(os.path.join(test_dir, '3D_patterns_epoch_' + str(epoch) + '.png'))


def main(_):

    model = GridCell_multidir_3d(FLAGS)

    with tf.Session() as sess:

        if FLAGS.mode == "1":  # visualize learned patterns
            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.training_output_dir, 'gau_model', FLAGS.ckpt))
            print('Loading checkpoint {}.'.format(os.path.join(FLAGS.training_output_dir, 'gau_model', FLAGS.ckpt)))
            test_dir = os.path.join(FLAGS.testing_output_dir, 'test_for_patterns_visualization')
            print("Testing.... please check folder %s " % test_dir)
            visualize_3D_grid_cell(model, sess, test_dir)

        elif FLAGS.mode == "2":  # test path integral

            model.path_integral(FLAGS.test_num, project_to_point=FLAGS.project_to_point)

            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.training_output_dir, 'gau_model', FLAGS.ckpt))
            print('Loading checkpoint {}.'.format(os.path.join(FLAGS.training_output_dir, 'gau_model', FLAGS.ckpt)))

            data_generator_test = Data_Generator(max=FLAGS.place_size, to_use_3D_map=True, num_interval=model.num_interval)

            place_pair_test = data_generator_test.generate(FLAGS.num_testing_path_integral, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2, test=True)

            syn_path_dir_testing = os.path.join(FLAGS.testing_output_dir, 'testing_path_integral')

            tf.gfile.MakeDirs(syn_path_dir_testing)

            print("Testing.... please check folder %s " % syn_path_dir_testing)
            err = test_path_integral(model, sess, place_pair_test, test_dir=syn_path_dir_testing, visualize=True)

            print("error of path integral in %s testing cases: %s" % (str(FLAGS.num_testing_path_integral), str(err)))

        elif FLAGS.mode == "0":

            print('Start training 3D grid cells')
            train(model, sess, FLAGS.training_output_dir)

        else:

            return NotImplementedError


if __name__ == '__main__':
    tf.app.run()
