import tensorflow as tf
import numpy as np
import os
import argparse
import math
from gridcell import GridCell
from custom_ops import block_diagonal
from data_io import Data_Generator
from matplotlib import pyplot as plt
from utils import draw_heatmap_2D, draw_path_to_target, draw_path_to_target_gif
import itertools
from scipy import io


class Path_planning():
    def __init__(self, grid_cell_model, max_step=200, max_err=2.0, obstacle_type='dot'):
        self.model = grid_cell_model
        # build model
        self.start = tf.placeholder(shape=[2], dtype=tf.float32)
        self.target = tf.placeholder(shape=[2], dtype=tf.float32)
        self.obstacle_type = obstacle_type
        if obstacle_type == 'dot':
            self.obstacle = tf.placeholder(shape=[2], dtype=tf.float32)
        else:
            self.obstacle = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.max_step, self.max_err = max_step, max_err
        self.a = tf.placeholder(dtype=tf.float32)
        self.b = tf.placeholder(dtype=tf.float32)
        # self.path_planning(max_step, max_err)

    def path_planning(self, num_step):
        step = tf.constant(0)
        grid_start = self.model.get_grid_code(self.start)
        grid_target = self.model.get_grid_code(self.target)
        grid_obstacle = self.model.get_grid_code(self.obstacle)
        place_seq, _ = self.model.localization_model(self.model.weights_A, grid_start, self.model.grid_cell_dim)
        place_seq = tf.expand_dims(place_seq, axis=0)
        place_seq_point = tf.expand_dims(self.start, axis=0)
        # velocity = self.model.velocity2
        num_dir = 100
        theta = np.linspace(0, np.pi * 2, num_dir + 1)[:num_dir]
        r = 2.0
        velocity = np.zeros(shape=(num_dir, 2), dtype=np.float32)
        velocity[:, 0] = r * np.cos(theta)
        velocity[:, 1] = r * np.sin(theta)
        vel_list = []

        interval_length = 1.0 / (self.model.num_interval - 1)
        for t in range(num_step):
            vel_list.append(velocity * (t + 1))
        r = 1.0
        velocity2 = np.zeros(shape=(num_dir, 2), dtype=np.float32)
        velocity2[:, 0] = r * np.cos(theta)
        velocity2[:, 1] = r * np.sin(theta)
        vel_list.append(velocity2)
        vel_list = np.concatenate(vel_list, axis=0)
        num_vel = len(velocity)

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

            inner_pd1 = tf.reduce_sum(grid_target * grid_next_pool, axis=1)
            if self.obstacle_type == 'dot':
                inner_pd2 = tf.reduce_sum(grid_obstacle * grid_next_pool, axis=1)
                direction_pool = inner_pd1 - self.a * tf.pow(inner_pd2, self.b)
            else:
                inner_pd2 = tf.reduce_sum(
                    tf.expand_dims(grid_obstacle, axis=1) * tf.expand_dims(grid_next_pool, axis=0), axis=-1)
                direction_pool = inner_pd1 - tf.reduce_sum(self.a * tf.pow(inner_pd2, self.b), axis=0)
            place_next_pool, _ = self.model.localization_model(self.model.weights_A, grid_next_pool, self.model.grid_cell_dim)

            p_max = tf.reduce_max(tf.reshape(place_next_pool, [-1, self.model.place_dim]), axis=1)
            g_max = tf.reduce_max(grid_next_pool, axis=1)
            mask = p_max > 0.7
            # mask = tf.logical_and(p_max > 0.1)
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
                                                                                         tf.TensorShape([None, len(vel_list)]),
                                                                                         tf.TensorShape([None, len(vel_list), self.model.grid_cell_dim])])

        self.place_seq, self.place_seq_point = place_seq, place_seq_point


def perform_path_planning(planning_model, sess, a, b, num_test=1000, max_step=40,
                          output_dir=None, test_dir_name='test_path_planning>20'):
    assert len(a) == len(b)
    num_param = len(a)
    output_dir = os.path.join(output_dir, test_dir_name)
    output_data_dir = os.path.join(output_dir, 'data')
    output_success_dir = os.path.join(output_dir, 'success')
    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    if not os.path.exists(output_success_dir):
        os.makedirs(output_success_dir)

    success = np.zeros(num_param, dtype=np.float32)
    success_step = np.zeros(num_param, dtype=np.float32)
    num_success = 0
    nbin = 4
    nvel = np.zeros(shape=nbin+1)
    count = np.zeros(shape=nbin+1)

    for tt in range(num_test):
        # Sample destination and starting point
        target_value = np.random.choice(planning_model.model.num_interval - 4, [100, 2]) + 2
        start_value = np.random.choice(planning_model.model.num_interval - 4, [100, 2]) + 2

        select_idx = np.where(np.sqrt(np.sum((target_value - start_value) ** 2, axis=1)) > 25)
        target_value, start_value = target_value[select_idx[0][0]], start_value[select_idx[0][0]]
        if planning_model.obstacle_type == 'dot':
            ratio = np.random.random() * 0.8 + 0.1
            obstacle_value = np.round(ratio * start_value + (1 - ratio) * target_value)
        elif planning_model.obstacle_type == 'line':
            center = (target_value + start_value) / 2
            obs_x = np.arange(max(center[0] - 8, 0), min(center[0] + 8, planning_model.model.num_interval))
            obstacle_value = np.stack((obs_x, np.ones(len(obs_x)) * center[1])).T
        elif planning_model.obstacle_type == 'rectangular':
            center = (target_value + start_value) / 2
            x, y = np.meshgrid(np.arange(center[0] - 3, center[0] + 3), np.arange(center[1] - 6, center[1] + 6))
            x, y = x.reshape(-1), y.reshape(-1)
            obstacle_value = np.stack((x, y)).T
        elif planning_model.obstacle_type == 'maze':
            while True:
                center = np.random.choice(planning_model.model.num_interval, size=(4, 2))
                x_list, y_list = [], []
                for idx in range(int(len(center) / 2)):
                    x, y = np.meshgrid(np.arange(np.fmax(0, center[idx, 0] - 3),
                                                 np.fmin(planning_model.model.num_interval, center[idx, 0] + 3)),
                                       np.arange(np.fmax(0, center[idx, 1] - 6),
                                                 np.fmin(center[idx, 1] + 6, planning_model.model.num_interval)))
                    x, y = x.reshape(-1), y.reshape(-1)
                    x_list.append(x), y_list.append(y)
                for idx in np.arange(int(len(center) / 2), len(center)):
                    x, y = np.meshgrid(np.arange(np.fmax(0, center[idx, 0] - 6),
                                                 np.fmin(planning_model.model.num_interval, center[idx, 0] + 6)),
                                       np.arange(np.fmax(0, center[idx, 1] - 3),
                                                 np.fmin(center[idx, 1] + 3, planning_model.model.num_interval)))
                    x, y = x.reshape(-1), y.reshape(-1)
                    x_list.append(x), y_list.append(y)
                x = np.concatenate(x_list)
                y = np.concatenate(y_list)
                obstacle_value = np.stack((x, y)).T
                if np.sum(np.sum((obstacle_value - start_value) ** 2, axis=1) == 0) == 0 and np.sum(np.sum((obstacle_value - target_value) ** 2, axis=1) == 0) == 0:
                    break
        elif planning_model.obstacle_type == 'bins':
            nbin = 4
            while True:
                center = np.random.uniform(low=8, high=planning_model.model.num_interval - 8, size=nbin)
                if sum(np.diff(np.sort(center)) < 2) == 0:
                    break
            hole_center = np.random.uniform(low=5, high=planning_model.model.num_interval - 5, size=nbin)
            x_list, y_list = [], []

            for i in range(nbin):
                x = np.hstack((np.arange(0, hole_center[i] - 3), np.arange(hole_center[i] + 3, planning_model.model.num_interval)))
                y = np.ones(len(x)) * center[i]
                x_list.append(x), y_list.append(y)
            x = np.concatenate(x_list)
            y = np.concatenate(y_list)
            obstacle_value = np.stack((x, y)).T
            start_value = np.array([np.random.uniform(low=3, high=planning_model.model.num_interval - 3),
                           np.random.uniform(low=1, high=8)])
            target_value = np.array([np.random.uniform(low=3, high=planning_model.model.num_interval - 3),
                                    np.random.uniform(low=planning_model.model.num_interval - 8, high=planning_model.model.num_interval - 1)])

        # Do path planning
        place_seq_list = []
        success_idx = 1
        for ii in range(num_param):
            feed_dict = {planning_model.start: start_value,
                         planning_model.target: target_value,
                         planning_model.obstacle: obstacle_value,
                         planning_model.a: a[ii],
                         planning_model.b: b[ii]}
            place_seq_point_value = sess.run(planning_model.place_seq_point, feed_dict=feed_dict)
            place_seq_list.append(place_seq_point_value)

            if planning_model.obstacle_type == 'dot':
                hit_obstacle = np.sum(np.sqrt(np.sum((place_seq_point_value - obstacle_value) ** 2, axis=1)) <= 2) > 0
            else:
                hit_obstacle = np.sum(np.sqrt(np.sum((np.tile(
                    np.expand_dims(place_seq_point_value, axis=1), (1, len(obstacle_value), 1)) -
                                                      obstacle_value) ** 2, axis=-1)) <= 2) > 0
            success_cond = len(place_seq_point_value) < max_step and not hit_obstacle
            if success_cond:
                success[ii] = success[ii] + 1
                success_step[ii] = success_step[ii] + len(place_seq_point_value)
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
            else:
                success_idx = 0

        if tt < 100:
            plt.figure(figsize=(5, 5))
            draw_path_to_target(planning_model.model.num_interval, place_seq_list,
                                target=target_value, obstacle=obstacle_value, a=a, b=b)
            plt.savefig(os.path.join(output_dir, 'test%02d.png' % tt))
            plt.close()
            data_dict = {'place_seq': place_seq_list, 'start': start_value, 'target': target_value,
                         'obstacle': obstacle_value, 'success': success_idx, 'a': a, 'b': b}
            io.savemat(os.path.join(output_data_dir, '%02d.mat' % tt), data_dict)

        if success_idx and num_success < 100:
            num_success = num_success + 1
            plt.figure(figsize=(5, 5))
            draw_path_to_target(planning_model.model.num_interval, place_seq_list,
                                target=target_value, obstacle=obstacle_value, a=a, b=b)
            plt.savefig(os.path.join(output_success_dir, '%02d.png' % tt))
            plt.close()
            data_dict = {'place_seq': place_seq_list, 'start': start_value, 'target': target_value,
                         'obstacle': obstacle_value, 'success': success_idx, 'a': a, 'b': b}
            io.savemat(os.path.join(output_success_dir, '%02d.mat' % tt), data_dict)

    nvel = nvel / count
    success_pro = success / float(num_test)
    success_step = success_step / success
    print(nvel)
    for ii in range(num_param):
        print('a=%02f, b=%02f: proportion of success %02f, average success step %02f' %
              (a[ii], b[ii], success_pro[ii], success_step[ii]))
    return success_pro, success_step


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
    parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.3], help='sd of gaussian kernel')
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
    parser.add_argument('--a', metavar='N', type=float, nargs='+', default=[0.5], help='annealing param')
    parser.add_argument('--b', metavar='N', type=float, nargs='+', default=[6.0], help='scaling param')

    parser.add_argument('--save_memory', type=bool, default=False, help='True if in testing mode')
    parser.add_argument('--plot', type=bool, default=True, help='True if in testing mode')

    # planning parameters
    parser.add_argument('--num_test', type=int, default=1000, help='Maximum number of steps')
    parser.add_argument('--planning_step', type=int, default=1, help='Maximum number of steps')
    parser.add_argument('--max_step', type=int, default=60, help='Maximum number of steps')
    parser.add_argument('--max_err', type=float, default=None, help='')
    parser.add_argument('--obstacle_type', type=str, default='dot', help='True if in testing mode')

    # utils
    parser.add_argument('--output_dir', type=str, default='con_G_s0.08_max40,3_t2',
                        help='Checkpoint path to load')
    parser.add_argument('--ckpt', type=str, default='model.ckpt-4499', help='Checkpoint path to load')
    parser.add_argument('--M_file', type=str, default='M.npy', help='Estimated M DILE')
    parser.add_argument('--test_dir_name', type=str, default='test_obstacle_t1', help='Estimated M file')
    parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')

    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    model = GridCell_multidir(FLAGS)
    planning_model = Path_planning(model, max_step=FLAGS.max_step, obstacle_type=FLAGS.obstacle_type)
    planning_model.path_planning(FLAGS.planning_step)

    with tf.Session() as sess:
        ckpt_file = os.path.join(FLAGS.output_dir, 'model', FLAGS.ckpt)
        # Load checkpoint
        assert FLAGS.ckpt is not None, 'no checkpoint provided.'
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('Loading checkpoint {}.'.format(ckpt_file))
        saver.restore(sess, ckpt_file)

        perform_path_planning(planning_model, sess, FLAGS.a, FLAGS.b, num_test=FLAGS.num_test, max_step=FLAGS.max_step,
                              output_dir=FLAGS.output_dir, test_dir_name=FLAGS.test_dir_name)


if __name__ == '__main__':
    tf.app.run()