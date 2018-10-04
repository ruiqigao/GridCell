from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from scipy.stats import norm
import sys
import cv2
import imageio
import matplotlib.cm as cm
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


def draw_heatmap(data, save_path, xlabels=None, ylabels=None):
    # data = np.clip(data, -0.05, 0.05)
    cmap = cm.get_cmap('rainbow', 1000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)

    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    plt.savefig(save_path)
    plt.close()


def draw_heatmap_2D(data, vmin=None, vmax=None):
    cmap = cm.get_cmap('rainbow', 1000)

    if vmin is None:
        vmax = data[0][0]
        vmin = data[0][0]
        for i in data:
            for j in i:
                if j > vmax:
                    vmax = j
                if j < vmin:
                    vmin = j
    plt.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')


def draw_path_to_target(place_len, place_seq, save_file=None, target=None, obstacle=None, a=None, b=None, col_scheme='single'):
    if save_file is not None:
        plt.figure(figsize=(5, 5))
    if type(place_seq) == list or np.ndim(place_seq) > 2:
        if col_scheme == 'rainbow':
            colors = cm.rainbow(np.linspace(0, 1, len(place_seq)))
        for i in range(len(place_seq)):
            p_seq = place_seq[i]
            color = colors[i] if col_scheme == 'rainbow' else 'darkcyan'
            label = 'a=%.2f, b=%d' % (a[i], b[i]) if (a is not None and b is not None) else ''
            plt.plot(p_seq[:, 1], place_len - p_seq[:, 0], 'o-', color=color,
                     ms=6, lw=2.0, label=label)
            if a is not None and len(a) > 1:
                plt.legend(loc='lower left', fontsize=12)
    else:
        if type(place_seq) == list:
            place_seq = place_seq[0]
        plt.plot(place_seq[:, 1], place_len - place_seq[:, 0], 'o-', ms=6, lw=2.0, color='darkcyan')

    if target is not None:
        plt.plot(target[1], place_len - target[0], '*', ms=12, color='r')
        if obstacle is not None:
            if np.ndim(obstacle) == 2:
                obstacle = obstacle.T
            plt.plot(obstacle[1], place_len - obstacle[0], 's', ms=8, color='dimgray')

    plt.xticks([])
    plt.yticks([])
    plt.xlim((0, place_len))
    plt.ylim((0, place_len))
    ax = plt.gca()
    ax.set_aspect(1)
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()

# def draw_path_to_target(place_len, place_seq, target=None, obstacle=None, col=(255, 0, 0)):
#     place_seq = np.round(place_seq).astype(int)
#     cmap = cm.get_cmap('rainbow', 1000)
#
#     canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
#     if target is not None:
#         cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
#         cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
#     else:
#         cv2.circle(canvas, tuple(place_seq[-1]), 2, col, -1)
#     for i in range(len(place_seq) - 1):
#         cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i+1]), col, 1)
#
#     plt.imshow(np.swapaxes(canvas, 0, 1), interpolation='nearest', cmap=cmap, aspect='auto', vmin=canvas.min(), vmax=canvas.max())
#     return canvas


def draw_two_path(place_len, place_gt, place_pd):
    place_gt = np.round(place_gt).astype(int)
    place_pd = np.round(place_pd).astype(int)
    plt.plot(place_gt[:, 0], place_len - place_gt[:, 1], c='k', lw=2.5, label='Real Path')
    plt.plot(place_pd[:, 0], place_len - place_pd[:, 1], 'o:', c='r', lw=2.5, ms=5, label='Predicted Path')
    plt.xlim((0, place_len))
    plt.ylim((0, place_len))
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', fontsize=14)
    ax = plt.gca()
    ax.set_aspect(1)


def draw_path_integral(place_len, place_seq, col=(255, 0, 0)):
    place_seq = np.round(place_seq).astype(int)
    cmap = cm.get_cmap('rainbow', 1000)

    canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
    if target is not None:
        cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
    else:
        cv2.circle(canvas, tuple(place_seq[-1]), 2, col, -1)
    for i in range(len(place_seq) - 1):
        cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i+1]), col, 1)

    plt.imshow(np.swapaxes(canvas, 0, 1), interpolation='nearest', cmap=cmap, aspect='auto')
    return canvas


def draw_path_to_target_gif(file_name, place_len, place_seq, target, col=(255, 0, 0)):
    cmap = cm.get_cmap('rainbow', 1000)
    canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
    cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)

    canvas_list = []
    canvas_list.append(canvas)
    for i in range(1, len(place_seq)):
        canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
        cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
        for j in range(i):
            cv2.line(canvas, tuple(place_seq[j]), tuple(place_seq[j+1]), col, 1)
        canvas_list.append(canvas)

    imageio.mimsave(file_name, canvas_list, 'GIF', duration=0.3)


def mu_to_map_old(mu, num_interval, max=1.0):
    if len(mu.shape) == 1:
        map = np.zeros([num_interval, num_interval], dtype=np.float32)
        map[mu[0], mu[1]] = max
    elif len(mu.shape) == 2:
        map = np.zeros([mu.shape[0], num_interval, num_interval], dtype=np.float32)
        for i in range(len(mu)):
            map[i, mu[i, 0], mu[i, 1]] = 1.0

    return map


def mu_to_map(mu, num_interval):
    mu = mu / float(num_interval)
    if len(mu.shape) == 1:
        discretized_x = np.expand_dims(np.linspace(0, 1, num=num_interval), axis=1)
        max_pdf = pow(norm.pdf(0, loc=0, scale=0.02), 2)
        vec_x_before = norm.pdf(discretized_x, loc=mu[0], scale=0.02)
        vec_y_before = norm.pdf(discretized_x, loc=mu[1], scale=0.02)
        map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
    elif len(mu.shape) == 2:
        map_list = []
        max_pdf = pow(norm.pdf(0, loc=0, scale=0.005), 2)
        for i in range(len(mu)):
            discretized_x = np.expand_dims(np.linspace(0, 1, num=num_interval), axis=1)
            vec_x_before = norm.pdf(discretized_x, loc=mu[i, 0], scale=0.005)
            vec_y_before = norm.pdf(discretized_x, loc=mu[i, 1], scale=0.005)
            map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
            map_list.append(map)
        map = np.stack(map_list, axis=0)

    return map


def generate_vel_list(max_vel, min_vel):
    vel_list = []
    max_vel_int = int(np.ceil(max_vel) + 1)
    for i in range(0, max_vel_int):
        for j in range(0, max_vel_int):
            if np.sqrt(i ** 2 + j ** 2) <= max_vel and np.sqrt(i ** 2 + j ** 2) >= min_vel:
                vel_list.append(np.array([i, j]))
                if i > 0:
                    vel_list.append(np.array([-i, j]))
                if j > 0:
                    vel_list.append(np.array([i, -j]))
                if i > 0 and j > 0:
                    vel_list.append(np.array([-i, -j]))
    vel_list = np.stack(vel_list)

    return vel_list