import tensorflow as tf
import numpy as np
import os
import argparse
import math
from gridcell_multidir_3d import GridCell_multidir_3d
from custom_ops import block_diagonal
from data_io import Data_Generator
from matplotlib import pyplot as plt
from utils import draw_heatmap_2D, draw_3D_path_to_target, draw_path_to_target_gif
import itertools


class Path_planning_3D():
    def __init__(self, grid_cell_model, max_step=80, max_err=1):
        self.model = grid_cell_model
        # build model
        self.start = tf.placeholder(shape=[3], dtype=tf.float32)
        self.target = tf.placeholder(shape=[3], dtype=tf.float32)
        self.max_step, self.max_err = max_step, max_err

        # self.path_planning(max_step, max_err)

    def path_planning(self, num_step, num_dir):
        step = tf.constant(0)
        grid_start = self.model.get_grid_code(self.start)
        grid_target = self.model.get_grid_code(self.target)
        place_seq, _ = self.model.localization_model(self.model.weights_A, grid_start, self.model.grid_cell_dim)
        place_seq = tf.expand_dims(place_seq, axis=0)
        place_seq_point = tf.expand_dims(self.start, axis=0)
        # velocity = self.model.velocity2

        theta = np.linspace(0, np.pi, num_dir + 1)[:num_dir]
        omega = np.linspace(0, 2*np.pi, num_dir + 1)[:num_dir]
        r = 2.0
        velocity = np.zeros(shape=(num_dir ** 2, 3), dtype=np.float32)

        index = 0
        for i_theta in theta:
            for i_omega in omega:
               velocity[index, 0] = r * np.sin(i_theta) * np.cos(i_omega)
               velocity[index, 1] = r * np.sin(i_theta) * np.sin(i_omega)
               velocity[index, 2] = r * np.cos(i_theta)
               index = index + 1

        num_vel = len(velocity)
        vel_list = []

        interval_length = 1.0 / (self.model.num_interval - 1)

        for t in range(num_step):
            vel_list.append(velocity * (t + 1))
        r = 1.0
        velocity2 = np.zeros(shape=(num_dir ** 2, 3), dtype=np.float32)

        index = 0
        for i_theta in theta:
            for i_omega in omega:
               velocity2[index, 0] = r * np.sin(i_theta) * np.cos(i_omega)
               velocity2[index, 1] = r * np.sin(i_theta) * np.sin(i_omega)
               velocity2[index, 2] = r * np.cos(i_theta)
               index = index + 1
                
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
                                                                                         tf.TensorShape([None, self.model.num_interval, self.model.num_interval, self.model.num_interval]),
                                                                                         tf.TensorShape([None, 3]),
                                                                                         tf.TensorShape([None, num_vel * (num_step + 1)]),
                                                                                         tf.TensorShape([None, num_vel * (num_step + 1), self.model.grid_cell_dim])])

        self.place_seq, self.place_seq_point = place_seq, place_seq_point


def perform_path_planning(planning_model, sess, num_test=1000, max_step=40,
                          output_dir=None, test_dir_name='test_path_planning'):

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
    a = 0.1

    for tt in range(num_test):
        # Sample destination and starting point
        target_value = np.random.choice(planning_model.model.num_interval - 4, [100, 3]) + 2
        start_value = np.random.choice(planning_model.model.num_interval - 4, [100, 3]) + 2

        select_idx = np.where(np.sqrt(np.sum((target_value - start_value) ** 2, axis=1)) > 20)

        target_value, start_value = target_value[select_idx[0][0]], start_value[select_idx[0][0]]

        # Do path planning
        feed_dict = {planning_model.start: start_value, planning_model.target: target_value}
        place_seq_value, place_seq_point_value, grid_next_pool, grid_list = sess.run([planning_model.place_seq, planning_model.place_seq_point, planning_model.grid_next_pool, planning_model.grid_code_list], feed_dict=feed_dict)

        if len(place_seq_value) < max_step:
            success = success + 1
            success_step = success_step + len(place_seq_value)

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

            draw_3D_path_to_target(planning_model.model.num_interval, place_seq_point_value, target_value)
            plt.savefig(os.path.join(output_dir, 'test%02d.png' % tt))
            plt.close()

    nvel = nvel / count
    success_pro = success / float(num_test)
    success_step = success_step / float(success)
    print(nvel)
    print('Proportion of success %02f, average success step %02f' % (success_pro, success_step))
    return success_pro, success_step


def main(_):
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate for descriptor')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')

    # simulated data parameters
    parser.add_argument('--to_use_3D_map', type=bool, default=True, help='to use 3D map or not')
    parser.add_argument('--place_size', type=float, default=1.0, help='Size of the square place')
    parser.add_argument('--max_vel1', type=float, default=39, help='maximum of velocity in loss1')
    parser.add_argument('--min_vel1', type=float, default=1, help='minimum of velocity in loss1')
    parser.add_argument('--max_vel2', type=float, default=3, help='maximum of velocity in loss2')
    parser.add_argument('--min_vel2', type=float, default=1, help='minimum of velocity in loss2')
    parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.3], help='sd of gaussian kernel')
    parser.add_argument('--num_data', type=int, default=30000, help='Number of simulated data points')

    # model parameters
    parser.add_argument('--place_dim', type=int, default=64000, help='Dimensions of place, should be N^3 or N^2')
    parser.add_argument('--num_group', type=int, default=10, help='Number of groups of grid cells')
    parser.add_argument('--block_size', type=int, default=6, help='Size of each block')
    parser.add_argument('--lamda', type=float, default=0.1, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--lamda2', type=float, default=1, help='Hyper parameter to balance two loss terms')
    parser.add_argument('--motion_type', type=str, default='continuous', help='True if in testing mode')
    parser.add_argument('--num_step', type=int, default=1, help='Number of steps in path integral')
    parser.add_argument('--GandE', type=float, default=0, help='1: Gaussian kernel; 0: Exponential kernel')
    parser.add_argument('--save_memory', type=bool, default=True, help='True if in testing mode')

    # planning parameters
    parser.add_argument('--num_test', type=int, default=1000, help='Maximum number of steps')
    parser.add_argument('--num_dir', type=int, default=90, help='number of directions to search')
    parser.add_argument('--planning_step', type=int, default=1, help='Maximum number of steps') #10
    parser.add_argument('--max_step', type=int, default=80, help='Maximum number of steps')
    parser.add_argument('--max_err', type=float, default=None, help='')

    # utils
    parser.add_argument('--training_output_dir', type=str, default='training_result',
                        help='Checkpoint path to load the model')
    parser.add_argument('--testing_output_dir', type=str, default='testing_result',
                        help='The output directory for saving testing results')

    parser.add_argument('--ckpt', type=str, default='model.ckpt-7999', help='Checkpoint name to load')
    parser.add_argument('--M_file', type=str, default='M.npy', help='Estimated M DILE')
    parser.add_argument('--test_dir_name', type=str, default='test_path_planning', help='name of folder for output')
    parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')

    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    
    model = GridCell_multidir_3d(FLAGS)
    
    planning_model = Path_planning_3D(model, FLAGS.max_step)
    planning_model.path_planning(FLAGS.planning_step, FLAGS.num_dir)

    with tf.Session() as sess:
        ckpt_file = os.path.join(FLAGS.training_output_dir, 'exp_model', FLAGS.ckpt)
        # Load checkpoint
        assert FLAGS.ckpt is not None, 'no checkpoint provided.'
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('Loading checkpoint {}.'.format(ckpt_file))
        saver.restore(sess, ckpt_file)

        print("Testing.... please check folder %s " % os.path.join(FLAGS.testing_output_dir, FLAGS.test_dir_name))
        perform_path_planning(planning_model, sess, num_test=FLAGS.num_test, max_step=FLAGS.max_step,
                              output_dir=FLAGS.testing_output_dir, test_dir_name=FLAGS.test_dir_name)


if __name__ == '__main__':
    tf.app.run()