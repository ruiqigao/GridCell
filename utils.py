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
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

def draw_pts(pts, clr, cmap, ax=None,sz=20):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        # ax.view_init(-45,-64)
        ax.view_init(15, -64)
    else:
        ax.cla()
    pts -= np.mean(pts,axis=0) #demean

    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)
    #for i, p in enumerate(pts):
    #    ax.text(p[0], p[1], p[2], str(i+1))
    if cmap is None and clr is not None:
        assert(np.all(clr.shape==pts.shape))
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _,clr,_ = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            # depthshade=False,
            edgecolors=(0.5, 0.5, 0.5)
        )

    #ax.set_axis_off()
    ax.set_facecolor("white")
    return ax,sct

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

def draw_3D_path_to_target(place_len, place_seq, target=None, obstacle=None, a=None, b=None):
    if type(place_seq) == list:
        start = place_seq[0][0]
    else:
        start = place_seq[0]

    if type(place_seq) == list:
        assert a is not None and b is not None
        #colors = cm.rainbow(np.linspace(0, 1, len(place_seq)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(place_seq)):

            p_seq = place_seq[i]

            #ax.plot(p_seq[:, 0], p_seq[:, 1], p_seq[:, 2], color="blue", label='a=%.2f, b=%d' % (a[i], b[i]))
            ax.plot(p_seq[:, 0], p_seq[:, 1], p_seq[:, 2], color="blue", label="planned path")
            ax.scatter(p_seq[0, 0], p_seq[0, 1], p_seq[0, 2], color="black", marker='o', label="origin")
            #plt.plot(p_seq[:, 1], place_len - p_seq[:, 0], 'o-', color=colors[i],
            #         ms=3, lw=2.0, label='a=%.2f, b=%d' % (a[i], b[i]))
            # ax.legend(loc='lower left', fontsize=12)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(place_seq[:, 0], place_seq[:, 1], place_seq[:, 2], color="blue", label="planned path")

        #plt.plot(place_seq[:, 1], place_len - place_seq[:, 0], 'o-', ms=3, lw=2.0)
        ax.scatter(place_seq[0, 0], place_seq[0, 1], place_seq[0, 2], color="black", marker='o', label="origin")

    if target is not None:
        # plt.plot(start[1], place_len - start[0], 'o', )
        ax.scatter(target[0], target[1], target[2], color="red", marker='x', label="target")
        #plt.plot(target[1], place_len - target[0], 'o', ms=10, color='dodgerblue')

        if obstacle is not None:
            if obstacle.size == 3:
                ax.scatter(obstacle[0], obstacle[1], obstacle[2], s=55, marker='s', color='y', label="obstacle")
                
            else:
                ax.scatter(obstacle[:, 0], obstacle[:, 1], obstacle[:, 2], s=55, marker='s', color='y',
                           label="obstacle")
                  
    ax.legend()
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    

    #plt.xticks([])
    #plt.yticks([])
    #plt.xlim((0, place_len))
    #plt.ylim((0, place_len))
    #ax = plt.gca()
    #ax.set_aspect(1)

def draw_path_to_target(place_len, place_seq, target=None, obstacle=None, a=None, b=None):
    if type(place_seq) == list:
        start = place_seq[0][0]
    else:
        start = place_seq[0]

    if type(place_seq) == list:
        assert a is not None and b is not None
        colors = cm.rainbow(np.linspace(0, 1, len(place_seq)))
        for i in range(len(place_seq)):
            p_seq = place_seq[i]
            plt.plot(p_seq[:, 1], place_len - p_seq[:, 0], 'o-', color=colors[i],
                     ms=3, lw=2.0, label='a=%.2f, b=%d' % (a[i], b[i]))
            plt.legend(loc='lower left', fontsize=12)
    else:
        plt.plot(place_seq[:, 1], place_len - place_seq[:, 0], 'o-', ms=3, lw=2.0)

    if target is not None:
        # plt.plot(start[1], place_len - start[0], 'o', )
        plt.plot(target[1], place_len - target[0], 'o', ms=10, color='dodgerblue')
        if obstacle is not None:
            plt.plot(obstacle[1], place_len - obstacle[0], '*', ms=10, color='r')

    plt.xticks([])
    plt.yticks([])
    plt.xlim((0, place_len))
    plt.ylim((0, place_len))
    ax = plt.gca()
    ax.set_aspect(1)


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
        cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i + 1]), col, 1)

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
            cv2.line(canvas, tuple(place_seq[j]), tuple(place_seq[j + 1]), col, 1)
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


def generate_vel_list_3d(max_vel, min_vel):
    vel_list = []
    max_vel_int = int(np.ceil(max_vel) + 1)
    for i in range(0, max_vel_int):
        for j in range(0, max_vel_int):
            for k in range(0, max_vel_int):
                if np.sqrt(i ** 2 + j ** 2 + k ** 2) <= max_vel and np.sqrt(i ** 2 + j ** 2 + k ** 2) >= min_vel:
                    vel_list.append(np.array([i, j, k]))
                    if i > 0:
                        vel_list.append(np.array([-i, j, k]))
                    if j > 0:
                        vel_list.append(np.array([i, -j, k]))
                    if k > 0:
                        vel_list.append(np.array([i, j, -k]))
                    if i > 0 and j > 0:
                        vel_list.append(np.array([-i, -j, k]))
                    if i > 0 and k > 0:
                        vel_list.append(np.array([-i, j, -k]))
                    if j > 0 and k > 0:
                        vel_list.append(np.array([i, -j, -k]))
                    if i > 0 and j > 0 and k > 0:
                        vel_list.append(np.array([-i, -j, -k]))

    vel_list = np.stack(vel_list)

    return vel_list
