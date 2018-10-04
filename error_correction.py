import tensorflow as tf
import numpy as np
import os
import argparse
import math
from gridcell_multidir import GridCell_multidir
from custom_ops import block_diagonal
from data_io import Data_Generator
from matplotlib import pyplot as plt
from utils import *
import itertools
from matplotlib import cm


class Error_correction():
    def __init__(self, grid_cell_model):
        self.model = grid_cell_model
        self.grid_cell_dim = grid_cell_model.grid_cell_dim
        self.motion_type = grid_cell_model.motion_type
        self.num_group = grid_cell_model.num_group

    def path_integral(self, test_num, noise_type='gaussian', sigma=0.1, dropout=0.3, project_to_point=False):
        # build testing model
        self.place_init_test = tf.placeholder(shape=[2], dtype=tf.float32)
        if self.motion_type == 'continuous':
            self.vel2_test = tf.placeholder(shape=[test_num, 2], dtype=tf.float32)
        else:
            self.vel2_test = tf.placeholder(shape=[test_num], dtype=tf.float32)

        place_seq_pd = []
        place_seq_pd_pt = []
        place_seq_pd_gp = []
        grid_code = self.model.get_grid_code(self.place_init_test)

        for step in range(test_num):
            # current_M = self.construct_motion_matrix(self.vel2_test[step], reuse=tf.AUTO_REUSE)
            if project_to_point is True and step > 0:
                grid_code = self.model.get_grid_code(tf.cast(place_seq_pd_pt[-1], tf.float32))

            M = self.model.construct_motion_matrix(self.vel2_test[step], reuse=tf.AUTO_REUSE)
            grid_code = self.model.motion_model(M, grid_code)

            # add noise to the grid code
            if noise_type == 'gaussian':
                grid_code = grid_code + tf.random_normal(shape=[self.grid_cell_dim], stddev=sigma)
            elif noise_type == 'dropout':
                idx = tf.cast(tf.random_shuffle(np.arange(self.grid_cell_dim)), tf.float32)
                mask = idx < self.grid_cell_dim * (1 - dropout)
                grid_code = grid_code * tf.cast(mask, tf.float32)

            place_pd, place_pd_pt = self.model.localization_model(self.model.weights_A, grid_code, self.model.grid_cell_dim, pd_pt=True)
            # place_pd_idx = tf.argmax(tf.reshape(place_pd, [-1]))
            # place_pd_pt = tf.cast(tf.transpose([tf.floordiv(place_pd_idx, self.num_interval),
            #                                     tf.mod(place_pd_idx, self.num_interval)]), tf.int32)
            place_seq_pd.append(place_pd)
            place_seq_pd_pt.append(place_pd_pt)

            place_seq_pd_gp_list = []
            for gp in range(self.num_group):
                gp_id = slice(gp * self.model.block_size, (gp + 1) * self.model.block_size)
                place_pd_gp, _ = self.model.localization_model(
                    tf.nn.l2_normalize(self.model.weights_A[:, :, gp_id], axis=-1), tf.nn.l2_normalize(grid_code[gp_id]),
                    self.model.block_size)
                place_seq_pd_gp_list.append(place_pd_gp)
            place_seq_pd_gp.append(tf.stack(place_seq_pd_gp_list))

        self.place_seq_pd, self.place_seq_pd_pt, self.place_seq_pd_gp = \
            tf.stack(place_seq_pd), tf.stack(place_seq_pd_pt), tf.stack(place_seq_pd_gp)

    def path_planning(self, num_step, noise_type='gaussian', sigma=0.1, dropout=0.3, project_to_point=False):
        self.start = tf.placeholder(shape=[2], dtype=tf.float32)
        self.target = tf.placeholder(shape=[2], dtype=tf.float32)
        self.max_step, self.max_err = 40, 2
        step = tf.constant(0)
        grid_start = self.model.get_grid_code(self.start)
        self.grid_target = self.model.get_grid_code(self.target)
        place_seq, _ = self.model.localization_model(self.model.weights_A, grid_start, self.model.grid_cell_dim)
        place_seq = tf.expand_dims(place_seq, axis=0)
        place_seq_point = tf.expand_dims(self.start, axis=0)
        # velocity = self.model.velocity2
        num_dir = 100
        theta = np.linspace(-np.pi, np.pi, num_dir + 1)[:num_dir]
        r = 2.0
        velocity = np.zeros(shape=(num_dir, 2), dtype=np.float32)
        velocity[:, 0] = r * np.cos(theta)
        velocity[:, 1] = r * np.sin(theta)
        num_vel = len(velocity)
        vel_list = []

        interval_length = 1.0 / (self.model.num_interval - 1)
        r = 1.0
        for t in range(num_step):
            vel_list.append(velocity * (t + 1))
        velocity2 = np.zeros(shape=(num_dir, 2), dtype=np.float32)
        velocity2[:, 0] = r * np.cos(theta)
        velocity2[:, 1] = r * np.sin(theta)
        vel_list.append(velocity2)
        vel_list = np.concatenate(vel_list, axis=0)

        M = self.model.construct_motion_matrix(tf.cast(velocity * interval_length, tf.float32), reuse=tf.AUTO_REUSE)
        M2 = self.model.construct_motion_matrix(tf.cast(velocity2 * interval_length, tf.float32), reuse=tf.AUTO_REUSE)
        place_max = tf.zeros(shape=(1, len(vel_list)))

        grid_code = tf.tile(tf.expand_dims(grid_start, axis=0), [num_vel, 1])
        grid_next_pool = []
        for t in range(num_step):
            grid_code = self.model.motion_model(M, grid_code)
            grid_next_pool.append(grid_code)
        grid_code = tf.tile(tf.expand_dims(grid_start, axis=0), [num_vel, 1])
        grid_code = self.model.motion_model(M2, grid_code)
        grid_next_pool.append(grid_code)
        self.grid_next_pool = tf.concat(grid_next_pool, axis=0)
        grid_code_list = tf.expand_dims(self.grid_next_pool, axis=0)

        def cond(step, grid_current, place_seq, place_seq_point, place_max, grid_code_list):
            return tf.logical_and(step < self.max_step,
                                  tf.sqrt(tf.reduce_sum((tf.to_float(place_seq_point[-1] - self.target)) ** 2)) > self.max_err)

        def body(step, grid_current, place_seq, place_seq_point, place_max, grid_code_list):
            # grid_current = self.model.get_grid_code(place_seq_point[-1])
            if noise_type == 'gaussian':
                grid_current = grid_current + tf.random_normal(shape=[self.grid_cell_dim], stddev=sigma)
                grid_target = self.grid_target + tf.random_normal(shape=[self.grid_cell_dim], stddev=sigma)
            elif noise_type == 'dropout':
                idx = tf.cast(tf.random_shuffle(np.arange(self.grid_cell_dim)), tf.float32)
                mask = idx < self.grid_cell_dim * (1 - dropout)
                grid_current = grid_current * tf.cast(mask, tf.float32)

                idx = tf.cast(tf.random_shuffle(np.arange(self.grid_cell_dim)), tf.float32)
                mask = idx < self.grid_cell_dim * (1 - dropout)
                grid_target = self.grid_target * tf.cast(mask, tf.float32)
            if project_to_point:
                _, place_pd_pt = self.model.localization_model(self.model.weights_A, grid_current,
                                                                      self.model.grid_cell_dim, pd_pt=True)
                grid_current = self.model.get_grid_code(tf.cast(place_pd_pt, tf.float32))

                _, target_pd_pt = self.model.localization_model(self.model.weights_A, grid_target,
                                                               self.model.grid_cell_dim, pd_pt=True)
                grid_target = self.model.get_grid_code(tf.cast(target_pd_pt, tf.float32))
            grid_code = tf.tile(tf.expand_dims(grid_current, axis=0), [num_vel, 1])
            grid_next_pool = []
            for t in range(num_step):
                grid_code = self.model.motion_model(M, grid_code)
                grid_next_pool.append(grid_code)

            grid_code = tf.tile(tf.expand_dims(grid_current, axis=0), [num_vel, 1])
            grid_code = self.model.motion_model(M2, grid_code)
            grid_next_pool.append(grid_code)
            grid_next_pool = tf.concat(grid_next_pool, axis=0)
            grid_code_list = tf.concat((grid_code_list, tf.expand_dims(grid_next_pool, axis=0)), axis=0)

            direction_pool = tf.reduce_sum(grid_target * grid_next_pool, axis=1)
            place_next_pool, _ = self.model.localization_model(self.model.weights_A, grid_next_pool, self.model.grid_cell_dim)

            p_max = tf.reduce_max(tf.reshape(place_next_pool, [-1, self.model.place_dim]), axis=1)
            g_max = tf.reduce_max(grid_next_pool, axis=1)
            mask = p_max > 0.5
            place_max = tf.concat([place_max, tf.expand_dims(p_max, axis=0)], axis=0)
            grid_next_pool, direction_pool = tf.boolean_mask(grid_next_pool, mask), tf.boolean_mask(direction_pool, mask)
            vel_pool = tf.boolean_mask(vel_list, mask)
            pick_idx = tf.argmax(direction_pool)

            grid_current = grid_next_pool[pick_idx]
            place_predict, _ = self.model.localization_model(self.model.weights_A, grid_current, self.model.grid_cell_dim)
            # place_point_predict = tf.cast(place_point_predict, tf.float32)
            place_pt = place_seq_point[-1] + tf.cast(vel_pool[pick_idx], tf.float32)

            place_seq = tf.concat([place_seq, tf.expand_dims(place_predict, axis=0)], axis=0)
            place_seq_point = tf.concat([place_seq_point, tf.expand_dims(place_pt, axis=0)], axis=0)
            return tf.add(step, 1), grid_current, place_seq, place_seq_point, place_max, grid_code_list

        _, self.grid_current, place_seq, place_seq_point, self.place_max, self.grid_code_list = tf.while_loop(cond, body, [step, grid_start, place_seq, place_seq_point, place_max, grid_code_list],
                                                                       shape_invariants=[step.get_shape(), grid_start.get_shape(),
                                                                                         tf.TensorShape([None, self.model.num_interval, self.model.num_interval]),
                                                                                         tf.TensorShape([None, 2]),
                                                                                         tf.TensorShape([None, num_vel * (num_step + 1)]),
                                                                                         tf.TensorShape([None, num_vel * (num_step + 1), self.model.grid_cell_dim])])

        self.place_seq, self.place_seq_point = place_seq, place_seq_point


def test_path_integral(model, sess, place_seq_test, visualize=False, test_dir=None, epoch=None):
    test_num = place_seq_test['seq'].shape[1] - 1
    err = np.zeros(shape=len(place_seq_test['seq']))
    # path integral
    place_min, place_max = 100, -100
    for i in range(len(place_seq_test['seq'])):
        feed_dict = {model.place_init_test: place_seq_test['seq'][i, 0],
                     model.vel2_test: place_seq_test['vel'][i]} if model.motion_type == 'continuous' \
            else {model.place_init_test: place_seq_test['seq'][i, 0],
                     model.vel2_test: place_seq_test['vel_idx'][i]}
        place_seq_pd_pt_value, place_seq_predict_value, place_seq_predict_gp_value = \
            sess.run([model.place_seq_pd_pt, model.place_seq_pd, model.place_seq_pd_gp], feed_dict=feed_dict)
        err[i] = np.mean(np.sqrt(np.sum((place_seq_test['seq'][i, 1:] - place_seq_pd_pt_value) ** 2, axis=1)))

        if place_seq_predict_value.min() < place_min:
            place_min = place_seq_predict_value.min()
        if place_seq_predict_value.max() > place_max:
            place_max = place_seq_predict_value.max()

        if visualize and i == 0:
            assert test_dir is not None
            if not tf.gfile.Exists(test_dir):
                tf.gfile.MakeDirs(test_dir)
            if epoch is not None:
                test_dir = os.path.join(test_dir, str(epoch))
                tf.gfile.MakeDirs(test_dir)

            cmap = cm.get_cmap('rainbow', 1000)
            num_gp_sqrt = max(int(np.ceil(np.sqrt(model.model.num_group))), 3)
            place_seq_pd_pt_value = np.vstack((place_seq_test['seq'][0, 0], place_seq_pd_pt_value[:test_num]))
            for j in range(test_num):
                plt.figure(figsize=(num_gp_sqrt, num_gp_sqrt + 1))
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 1)
                draw_path_to_target(model.model.num_interval, place_seq_test['seq'][0, :j+1])
                # draw_heatmap_2D(mu_to_map(place_pair_test['place_seq'][i], model.num_interval), vmin=0, vmax=1)
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 2)
                draw_path_to_target(model.model.num_interval, np.vstack((place_seq_test['seq'][0, 0], place_seq_pd_pt_value[:j])))
                plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 3)
                draw_heatmap_2D(place_seq_predict_value[j])
                for gp in range(model.model.num_group):
                    plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, gp + num_gp_sqrt + 1)
                    draw_heatmap_2D(place_seq_predict_gp_value[j, gp])
                plt.savefig(os.path.join(test_dir, str(j) + '.png'))
                plt.close()
            plt.figure(figsize=(5, 5))
            draw_two_path(model.model.num_interval, place_seq_test['seq'][i, :test_num + 1], place_seq_pd_pt_value)
            plt.savefig(os.path.join(test_dir, 'paths' + str(i) + '.png'))
            adict = {}
            adict['pt_pd'] = place_seq_pd_pt_value
            adict['pt_gt'] = place_seq_test['seq'][i]
            # savemat(os.path.join(FLAGS.output_dir, 'test', 'pt_seq' + str(i) + '.mat'), adict)
    err = np.mean(err)
    print(place_max, place_min)
    print(err)

    # visualized outcome
    return err


def perform_path_planning(planning_model, sess, start, target, max_step=40,
                          output_dir=None, test_dir_name='test_path_planning', plot=True):
    output_dir = os.path.join(output_dir, test_dir_name)
    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = 0
    success_step = 0
    nbin = 4
    nvel = np.zeros(shape=nbin+1)
    count = np.zeros(shape=nbin+1)
    num_test = len(start)
    place_seq_list = []

    for tt in range(num_test):
        start_value, target_value = start[tt], target[tt]
        # Sample destination and starting point
        # target_value = np.random.choice(planning_model.model.num_interval - 4, [100, 2]) + 2
        # start_value = np.random.choice(planning_model.model.num_interval - 4, [100, 2]) + 2
        # select_idx = np.where(np.sqrt(np.sum((target_value - start_value) ** 2, axis=1)) > 20)
        # target_value, start_value = target_value[select_idx[0][0]], start_value[select_idx[0][0]]

        # Do path planning
        feed_dict = {planning_model.start: start_value, planning_model.target: target_value}
        place_seq_value, place_seq_point_value, grid_next_pool, grid_list = sess.run([planning_model.place_seq,
                                                                                      planning_model.place_seq_point,
                                                                                      planning_model.grid_next_pool,
                                                                                      planning_model.grid_code_list],
                                                                                     feed_dict=feed_dict)

        if len(place_seq_value) < max_step:
            success = success + 1
            success_step = success_step + len(place_seq_value)
            # if success < 100:
            #     if not os.path.exists(os.path.join(output_dir, 'gif')):
            #         os.mkdir(os.path.join(output_dir, 'gif'))
            #     file_name = os.path.join(output_dir, 'gif', 'success%02d.gif' % success)
            #     draw_path_to_target_gif(file_name, planning_model.model.num_interval, place_seq_point_value, target_value)

            vel_seq = np.diff(place_seq_point_value, axis=0)
            vel_seq = np.sqrt(np.sum(np.square(vel_seq), axis=1))
            nseq = len(vel_seq)
            bin_sz = int(np.floor(nseq / nbin))
            for i in range(nbin):
                nvel[i] = nvel[i] + np.sum(vel_seq[i * bin_sz: max((i+1) * bin_sz, nseq)])
                count[i] = count[i] + max((i+1) * bin_sz, nseq) - i * bin_sz
            nvel[-1] = nvel[-1] + vel_seq[nseq-1]
            count[-1] = count[-1] + 1

        if tt < 100:
            if plot:
                draw_path_to_target(planning_model.model.num_interval, place_seq_point_value, target=target_value,
                                    save_file=os.path.join(output_dir, 'test%02d.png' % tt))
            place_seq_list.append(place_seq_point_value)

    nvel = nvel / count
    success_pro = success / float(num_test)
    success_step = success_step / float(success)
    print(nvel)
    print('Proportion of success %02f, average success step %02f' % (success_pro, success_step))
    return place_seq_list


def main(_):
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate for descriptor')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')

    # simulated data parameters
    parser.add_argument('--place_size', type=float, default=1.0, help='Size of the square place')
    parser.add_argument('--max_vel1', type=float, default=39, help='maximum of velocity in loss1')
    parser.add_argument('--min_vel1', type=float, default=1, help='minimum of velocity in loss1')
    parser.add_argument('--max_vel2', type=float, default=3, help='maximum of velocity in loss2')
    parser.add_argument('--min_vel2', type=float, default=1, help='minimum of velocity in loss2')
    parser.add_argument('--num_data', type=int, default=30000, help='Number of simulated data points')

    # model parameters
    parser.add_argument('--place_dim', type=int, default=1600, help='Dimensions of place, should be N^2')
    parser.add_argument('--num_group', type=int, default=16, help='Number of groups of grid cells')
    parser.add_argument('--block_size', type=int, default=6, help='Size of each block')
    parser.add_argument('--lamda', type=float, default=0.1, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--lamda2', type=float, default=1, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--motion_type', type=str, default='continuous', help='True if in testing mode')
    parser.add_argument('--num_step', type=int, default=1, help='Number of steps in path integral')
    parser.add_argument('--GandE', type=float, default=1.0, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--save_memory', type=bool, default=False, help='True if in testing mode')

    # error correction parameter
    parser.add_argument('--test_type', type=str, default='normal', help='Maximum number of steps')
    parser.add_argument('--task', type=str, default='integral', help='True if in testing mode')
    parser.add_argument('--num_test', type=int, default=100, help='Maximum number of steps')
    parser.add_argument('--sigma', type=float, default=0.08, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--project', type=bool, default=False, help='True if in testing mode')
    parser.add_argument('--dropout', type=float, default=0.7, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--noise_type', type=str, default='gaussian', help='True if in testing mode')

    # integral parameter
    parser.add_argument('--integral_step', type=int, default=30, help='Number of steps in path integral')

    # planning parameters
    parser.add_argument('--max_step', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--max_err', type=float, default=None, help='')
    parser.add_argument('--planning_step', type=int, default=1, help='Maximum number of steps')

    # utils
    parser.add_argument('--output_dir', type=str, default='con_G_s0.08_max40,3_t1_rg1',
                        help='Checkpoint path to load')
    parser.add_argument('--ckpt', type=str, default='model.ckpt-5999', help='Checkpoint path to load')
    parser.add_argument('--M_file', type=str, default='M.npy', help='Estimated M DILE')
    parser.add_argument('--test_dir_name', type=str, default='test_err_intergral', help='Estimated M file')
    parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')

    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    model = GridCell_multidir(FLAGS)
    error_correction_model = Error_correction(model)

    test_dir = os.path.join(FLAGS.output_dir, FLAGS.test_dir_name)

    with tf.Session() as sess:
        assert FLAGS.ckpt is not None, 'no checkpoint provided.'
        ckpt_file = os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt)

        if FLAGS.task == 'integral':
            if FLAGS.test_type == 'normal':
                error_correction_model.path_integral(FLAGS.integral_step, project_to_point=FLAGS.project,
                                                     sigma=FLAGS.sigma, dropout=FLAGS.dropout,
                                                     noise_type=FLAGS.noise_type)
                # Load checkpoint
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                print('Loading checkpoint {}.'.format(ckpt_file))
                saver.restore(sess, ckpt_file)

                data_generator_test = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval)
                place_pair_test = data_generator_test.generate(FLAGS.num_test, velocity=model.velocity2, num_step=FLAGS.integral_step, dtype=2,
                                                               test=True)
                err = test_path_integral(error_correction_model, sess, place_pair_test, visualize=True, test_dir=test_dir)
                print('%s %f' % (FLAGS.output_dir, np.mean(err)))
            elif FLAGS.test_type == 'comparison':
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                print('Loading checkpoint {}.'.format(ckpt_file))
                saver.restore(sess, ckpt_file)

                data_generator_test = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval)
                place_pair_test = data_generator_test.generate(FLAGS.num_test, velocity=model.velocity2,
                                                               num_step=FLAGS.integral_step, dtype=2,
                                                               test=True)
                err = test_path_integral(error_correction_model, sess, place_pair_test, visualize=True,
                                         test_dir=test_dir)

        elif FLAGS.task == 'planning':
            error_correction_model.path_planning(FLAGS.planning_step, project_to_point=FLAGS.project,
                                                 sigma=FLAGS.sigma, dropout=FLAGS.dropout, noise_type=FLAGS.noise_type)
            # Load checkpoint
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print('Loading checkpoint {}.'.format(ckpt_file))
            saver.restore(sess, ckpt_file)
            target_value = np.random.choice(error_correction_model.model.num_interval - 4, [FLAGS.num_test * 10, 2]) + 2
            start_value = np.random.choice(error_correction_model.model.num_interval - 4, [FLAGS.num_test * 10, 2]) + 2
            select_idx = np.where(np.sqrt(np.sum((target_value - start_value) ** 2, axis=1)) > 20)[0]
            target_value, start_value = target_value[select_idx[:FLAGS.num_test]], start_value[
                select_idx[:FLAGS.num_test]]

            perform_path_planning(error_correction_model, sess, start_value, target_value, max_step=FLAGS.max_step,
                                  output_dir=FLAGS.output_dir,
                                  test_dir_name='%s_t%d' % (FLAGS.test_dir_name, FLAGS.planning_step), plot=True)


if __name__ == '__main__':
    tf.app.run()