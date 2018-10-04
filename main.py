from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from gridcell import GridCell
from gridcell_multidir import GridCell_multidir
import os
from data_io import Data_Generator
from utils import *
from path_planning import Path_planning, perform_path_planning
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import argparse
from scipy.io import savemat

# tf.set_random_seed(1234)
# np.random.seed(0)

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--batch_size', type=int, default=30000, help='Batch size of training images')
parser.add_argument('--num_epochs', type=int, default=6000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate for descriptor')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')

# simulated data parameters
parser.add_argument('--place_size', type=float, default=1.0, help='Size of the square place')
parser.add_argument('--max_vel1', type=float, default=39, help='maximum of velocity in loss1')
parser.add_argument('--min_vel1', type=float, default=1, help='minimum of velocity in loss1')
parser.add_argument('--max_vel2', type=float, default=3, help='maximum  of velocity in loss2')
parser.add_argument('--min_vel2', type=float, default=1, help='minimum of velocity in loss2')
parser.add_argument('--num_data', type=int, default=30000, help='Number of simulated data points')
parser.add_argument('--dtype1', type=int, default=1, help='type of loss1')

# model parameters
parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.08], help='sd of gaussian kernel')
parser.add_argument('--place_dim', type=int, default=1600, help='Dimensions of place, should be N^2')
parser.add_argument('--num_group', type=int, default=16, help='Number of groups of grid cells')
parser.add_argument('--block_size', type=int, default=6, help='Size of each block')
parser.add_argument('--iter', type=int, default=0, help='Number of iter')
parser.add_argument('--lamda', type=float, default=0.1, help='Hyper parameter to balance two loss terms')
parser.add_argument('--GandE', type=float, default=1.0, help='Hyper parameter to balance two loss terms')
parser.add_argument('--lamda2', type=float, default=5000, help='Hyper parameter to balance two loss terms')
parser.add_argument('--motion_type', type=str, default='continuous', help='True if in testing mode')
parser.add_argument('--num_step', type=int, default=1, help='Number of steps in path integral')
parser.add_argument('--save_memory', type=bool, default=False, help='True if in testing mode')

# utils train
parser.add_argument('--output_dir', type=str, default='test', help='The output directory for saving results')
parser.add_argument('--err_dir', type=str, default=None, help='The output directory for saving results')
parser.add_argument('--log_file', type=str, default='con_test.txt', help='The output directory for saving results')
parser.add_argument('--log_step', type=int, default=500, help='Number of mini batches to save output results')

# utils test
parser.add_argument('--test_num', type=int, default=30, help='Number of testing steps used in path integral')
parser.add_argument('--test', type=bool, default=False, help='True if in testing mode')
parser.add_argument('--test_path', type=bool, default=False, help='True if in testing path integral mode')
parser.add_argument('--project_to_point', type=bool, default=False, help='True if in testing path integral mode')
parser.add_argument('--ckpt', type=str, default='model.ckpt-5999', help='Checkpoint path to load')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def train(model, sess, output_dir):
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

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

    data_generator = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval)

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
                                                                     model.loss2, model.reg, model.loss3, model.loss4,
                                                                     model.dp1, model.dp2, model.loss_update,
                                                                     model.apply_grads], feed_dict=feed_dict)[:8]

            # regularize weights
            if epoch > 4000:
                sess.run(model.norm_grads)

            loss1_avg.append(loss1)
            loss2_avg.append(loss2)
            reg_avg.append(reg)
            loss3_avg.append(loss3)
            loss4_avg.append(loss4)

        writer.add_summary(summary, epoch)
        writer.flush()
        if epoch % 10 == 0:
            loss1_avg, loss2_avg, loss3_avg, loss4_avg, reg_avg = np.mean(np.asarray(loss1_avg)), np.mean(np.asarray(loss2_avg)), \
                                                         np.mean(np.asarray(loss3_avg)), np.mean(np.asarray(loss4_avg)), \
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
            print('#{:s} Epoch #{:d}, train loss1: {:.4f}, train loss2: {:.4f}, reg: {:.4f}, val loss: {:.4f} time: {:.2f}s'
                  .format(output_dir, epoch, loss1_avg, loss2_avg, reg_avg, loss_val, end_time - start_time))
            # print('dp1: {:.4f}, dp2: {:.4f}'.format(np.sum(dp1), np.sum(dp2)))
            start_time = time.time()

        syn_dir = os.path.join(FLAGS.output_dir, 'syn')
        syn_path_dir = os.path.join(FLAGS.output_dir, 'syn_path')
        if epoch == 0 or (epoch + 1) % FLAGS.log_step == 0 or epoch == FLAGS.num_epochs - 1:
            saver.save(sess, "%s/%s" % (model_dir, 'model.ckpt'), global_step=epoch)
            if not tf.gfile.Exists(syn_dir):
                tf.gfile.MakeDirs(syn_dir)
            place_before, place_after, place_after_pd, place_before_infer, place_after_infer = \
                sess.run([model.place_start_gt, model.place_end_gt, model.place_end_pd, model.place_start_infer, model.place_end_infer], feed_dict=feed_dict)
            for ii in range(5):
                plt.figure(figsize=(3, 2))
                plt.subplot(2, 3, 1)
                draw_heatmap_2D(mu_to_map(place_before[ii], model.num_interval))
                plt.subplot(2, 3, 2)
                draw_heatmap_2D(mu_to_map(place_after[ii], model.num_interval))
                plt.subplot(2, 3, 4)
                draw_heatmap_2D(place_before_infer[ii])
                plt.subplot(2, 3, 5)
                draw_heatmap_2D(place_after_infer[ii])
                plt.subplot(2, 3, 6)
                draw_heatmap_2D(place_after_pd[ii])
                plt.savefig(os.path.join(syn_dir, str(ii) + '.png'))
                plt.close()

            test(model, sess, syn_dir, epoch, final_epoch=True, result_dir='./tune_results')

            place_seq_test = data_generator.generate(1, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2, test=True)
            test_path_integral(model, sess, place_seq_test, visualize=True, test_dir=syn_path_dir, epoch=epoch)

            place_seq_test = data_generator.generate(100, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2, test=True)
            err = test_path_integral(model, sess, place_seq_test)
            err = np.mean(err)
            if FLAGS.log_file is not None:
                with open(FLAGS.log_file, "a") as f:
                    print('%s %d %02f' % (output_dir, epoch, err), file=f)

        #
        #     # test path planning
        #     success_pro, success_step = perform_path_planning(planning_model, sess, output_dir=output_dir)
        #     print('Folder %s: proportion of success %02f, average success step %02f'
        #           % (output_dir, success_pro, success_step), file=open("score.txt", "a"))


def test_path_integral(model, sess, place_seq_test, visualize=False, test_dir=None, epoch=None):
    test_num = place_seq_test['seq'].shape[1] - 1
    err = np.zeros(shape=(len(place_seq_test['seq']), test_num))
    # path integral
    place_min, place_max = 100, -100
    for i in range(len(place_seq_test['seq'])):
        feed_dict = {model.place_init_test: place_seq_test['seq'][i, 0],
                     model.vel2_test: place_seq_test['vel'][i]} if model.motion_type == 'continuous' \
            else {model.place_init_test: place_seq_test['seq'][i, 0],
                     model.vel2_test: place_seq_test['vel_idx'][i]}
        place_seq_pd_pt_value, place_seq_predict_value, place_seq_predict_gp_value = \
            sess.run([model.place_seq_pd_pt, model.place_seq_pd, model.place_seq_pd_gp], feed_dict=feed_dict)
        err[i] = np.sqrt(np.sum((place_seq_test['seq'][i, 1:] - place_seq_pd_pt_value) ** 2, axis=1))

        if place_seq_predict_value.min() < place_min:
            place_min = place_seq_predict_value.min()
        if place_seq_predict_value.max() > place_max:
            place_max = place_seq_predict_value.max()

        if visualize:
            assert test_dir is not None
            if not tf.gfile.Exists(test_dir):
                tf.gfile.MakeDirs(test_dir)
            if epoch is not None:
                test_dir = os.path.join(test_dir, str(epoch))
                tf.gfile.MakeDirs(test_dir)

            cmap = cm.get_cmap('rainbow', 1000)
            num_gp_sqrt = max(int(np.ceil(np.sqrt(model.num_group))), 3)
            place_seq_pd_pt_value = np.vstack((place_seq_test['seq'][0, 0], place_seq_pd_pt_value[:test_num]))
            for j in range(test_num):
                plt.figure(figsize=(num_gp_sqrt, num_gp_sqrt + 1))
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 1)
                draw_path_to_target(model.num_interval, place_seq_test['seq'][0, :j+1])
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 2)
                draw_path_to_target(model.num_interval, np.vstack((place_seq_test['seq'][0, 0], place_seq_pd_pt_value[:j])))
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 3)
                draw_heatmap_2D(place_seq_predict_value[j], vmin=0, vmax=1)
                for gp in range(model.num_group):
                    plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, gp + num_gp_sqrt + 1)
                    draw_heatmap_2D(place_seq_predict_gp_value[j, gp], vmin=0, vmax=1)
                plt.savefig(os.path.join(test_dir, str(j) + '.png'))
                plt.close()
            plt.figure(figsize=(5, 5))
            draw_two_path(model.num_interval, place_seq_test['seq'][i, :test_num + 1], place_seq_pd_pt_value)
            plt.savefig(os.path.join(test_dir, 'paths' + str(i) + '.png'))
            adict = {}
            adict['pt_pd'] = place_seq_pd_pt_value
            adict['pt_gt'] = place_seq_test['seq'][i]
            # savemat(os.path.join(FLAGS.output_dir, 'test', 'pt_seq' + str(i) + '.mat'), adict)
    # err = np.mean(err)
    # err = np.mean(err, axis=0)
    print(place_max, place_min)

    # visualized outcome
    return err


def test(model, sess, test_dir, epoch=0, final_epoch=False, result_dir=None):
    weights_A_value = sess.run(model.weights_A)
    if not tf.gfile.Exists(test_dir):
        tf.gfile.MakeDirs(test_dir)
    np.save(os.path.join(test_dir, 'weights.npy'), weights_A_value)

    # print out A
    weights_A_value_transform = np.swapaxes(np.swapaxes(weights_A_value, 0, 2), 1, 2)
    # fig_sz = np.ceil(np.sqrt(len(weights_A_value_transform)))
    plt.figure(figsize=(model.block_size, model.num_group))
    for i in range(len(weights_A_value_transform)):
        weight_to_draw = weights_A_value_transform[i]
        plt.subplot(model.num_group, model.block_size, i + 1)
        draw_heatmap_2D(weight_to_draw, vmin=weight_to_draw.min(), vmax=weight_to_draw.max())
    plt.savefig(os.path.join(test_dir, 'A_curve_' + str(epoch) + '.png'))
    if final_epoch:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        plt.savefig(os.path.join(result_dir, FLAGS.output_dir + '.png'))
    plt.close()


def main(_):
    model = GridCell_multidir(FLAGS)

    with tf.Session() as sess:
        if FLAGS.test:
            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt))
            print('Loading checkpoint {}.'.format(os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt)))
            test_dir = os.path.join(FLAGS.output_dir, 'test')
            test(model, sess, test_dir)
        elif FLAGS.test_path:
            model.path_integral(FLAGS.test_num, project_to_point=FLAGS.project_to_point)
            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt))
            print('Loading checkpoint {}.'.format(os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt)))

            test_dir = os.path.join(FLAGS.output_dir, 'test')
            data_generator_test = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval)
            # place_seq_test = data_generator_test.generate(1, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2,
            #                                          test=True, visualize=True)
            # test_path_integral(model, sess, place_seq_test, visualize=True, test_dir=test_dir)
            place_pair_test = data_generator_test.generate(1000, velocity=model.velocity2, num_step=FLAGS.test_num, dtype=2, test=True)
            err = test_path_integral(model, sess, place_pair_test)
            if FLAGS.err_dir is not None:
                np.save(FLAGS.err_dir, err)

            err = np.mean(err)
            print('%s %f' % (FLAGS.output_dir, err))
            if FLAGS.log_file is not None:
                with open(FLAGS.log_file, "a") as f:
                    print('%s %02f' % (FLAGS.output_dir, err), file=f)
        else:
            train(model, sess, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
