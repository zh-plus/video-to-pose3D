# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from common.utils import read_video


def ckpt_time(ckpt=None, display=0, desc=''):
    if not ckpt:
        return time.time()
    else:
        if display:
            print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
        return time.time() - float(ckpt), time.time()


# def get_resolution(filename):
#     command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
#                '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
#     with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
#         for line in pipe.stdout:
#             w, h = line.decode().strip().split(',')
#             return int(w), int(h)
#
#
# def read_video(filename, fps=None, skip=0, limit=-1):
#     w, h = get_resolution(filename)
#
#     command = ['ffmpeg',
#                '-i', filename,
#                '-f', 'image2pipe',
#                '-pix_fmt', 'rgb24',
#                '-vsync', '0',
#                '-vcodec', 'rawvideo', '-']
#     if fps:
#         command.extend(['-r', str(fps)])
#
#     i = 0
#     with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
#         while True:
#             data = pipe.stdout.read(w * h * 3)
#             if not data:
#                 break
#             i += 1
#             if i > skip:
#                 yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
#             if i == limit:
#                 break


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


# def render_animation(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,
#                      limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
#     """
#     TODO
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     plt.ioff()
#     fig = plt.figure(figsize=(size * (1 + len(poses)), size))
#     # 2D
#     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
#     ax_in.get_xaxis().set_visible(False)
#     ax_in.get_yaxis().set_visible(False)
#     ax_in.set_axis_off()
#     ax_in.set_title('Input')
#
#     ax_3d = []
#     lines_3d = []
#     trajectories = []
#     radius = 1.7
#     for index, (title, data) in enumerate(poses.items()):
#         # 3D
#         ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         # set 长度范围
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_zlim3d([0, radius])
#         ax.set_ylim3d([-radius / 2, radius / 2])
#
#         # ax.set_aspect('equal')
#         ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])
#
#         # 坐标轴刻度
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 7.5
#         ax.set_title(title)  # , pad=35
#         ax_3d.append(ax)
#         lines_3d.append([])
#
#         # lxy add
#         ax.set_xlabel('X Label')
#         ax.set_ylabel('Y Label')
#         ax.set_zlabel('Z Label')
#
#         # 轨迹 is base on position 0
#         trajectories.append(data[:, 0, [0, 1]])  # only add x,y not z
#     poses = list(poses.values())
#
#     # ###################################################
#     #  xx = trajectories[0]
#     #  bad_points = []
#     #  for i in  range(len(xx)-1):
#     #  dis = xx[i+1] - xx[i]
#     #  value = abs(dis[0]) + abs(dis[1])
#     #  if value > 1e-06:
#     #  bad_points.append(i)
#     #  print(i, '      ',value)
#
#     # Decode video
#     if input_video_path is None:
#         # Black background
#         all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
#     else:
#         # Load video using ffmpeg
#         all_frames = []
#         for f in read_video(input_video_path, skip=input_video_skip):
#             all_frames.append(f)
#         effective_length = min(keypoints.shape[0], len(all_frames))
#         all_frames = all_frames[:effective_length]
#
#     if downsample > 1:
#         keypoints = downsample_tensor(keypoints, downsample)
#         all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
#         for idx in range(len(poses)):
#             poses[idx] = downsample_tensor(poses[idx], downsample)
#             trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
#         fps /= downsample
#
#     initialized = False
#     image = None
#     lines = []
#     points = None
#
#     if limit < 1:
#         limit = len(all_frames)
#     else:
#         limit = min(limit, len(all_frames))
#
#     # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
#     parents = skeleton.parents()
#
#     pbar = tqdm(total=limit)
#
#     def update_video(i):
#         nonlocal initialized, image, lines, points
#
#         for n, ax in enumerate(ax_3d):
#             # 对于 each video
#             ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
#             ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
#
#         # Update 2D poses
#         if not initialized:
#             image = ax_in.imshow(all_frames[i], aspect='equal')
#
#             for j, j_parent in enumerate(parents):
#                 # 每个 keypoint of each frame
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1]:
#                     color_pink = 'pink'
#                     if j == 1 or j == 2:
#                         color_pink = 'black'
#                     # 画图2D
#                     #  #  Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
#                     #  lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                     #  [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color=color_pink))
#
#                 col = 'red' if j in skeleton.joints_right() else 'black'
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     # 画图3D
#                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
#                                                [pos[j, 1], pos[j_parent, 1]],
#                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
#
#             # 一个 frame 的 scatter image
#             #  points = ax_in.scatter(*keypoints[i].T, 8, color='red', edgecolors='white', zorder=10)
#
#             initialized = True
#         else:
#             image.set_data(all_frames[i])
#
#             # 对于 each frame
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#
#                 #  # 画图2D
#                 #  if len(parents) == keypoints.shape[1]:
#                 #  lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                 #  [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
#
#                 # 3D plot
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]  # one frame key points
#                     lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
#                     lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
#                     lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
#
#             #  ax_3d.append(ax_3d[0])
#             # rotate the Axes3D
#             for angle in range(0, 360):
#                 # 仰角 方位角
#                 ax_3d[0].view_init(0, 90)
#
#             #  points.set_offsets(keypoints[i])
#
#         # print('finish one frame\t  {}/{}      '.format(i, limit))
#         pbar.update()
#         # sys.stdout.flush()
#
#     fig.tight_layout()
#
#     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
#     print(f'\nSaving {output}')
#     if output.endswith('.mp4'):
#         Writer = writers['ffmpeg']
#         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#         anim.save(output, writer=writer)
#     elif output.endswith('.gif'):
#         anim.save(output, dpi=80, writer='imagemagick')
#     else:
#         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#     plt.close()


def render_animation_my(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,
                        limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    # prevent wired error
    axes_3d = Axes3D(fig)

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])

        # ax.set_aspect('equal')
        ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 12.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, fps=25, skip=input_video_skip):
            all_frames.append(f)

        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    pbar = tqdm(total=limit)

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                #     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])

        pbar.update()

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000.0 / fps, repeat=False)

    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def render_animation_test(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport, limit=-1, downsample=1, size=6, input_video_frame=None,
                          input_video_skip=0, num=None):
    t0 = ckpt_time()
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    fig.add_subplot(121)
    plt.imshow(input_video_frame)
    # 3D
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=15., azim=azim)
    # set 长度范围
    radius = 1.7
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5

    # lxy add
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()

    pos = poses['Reconstruction'][-1]
    _, t1 = ckpt_time(t0, desc='1 ')
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        if len(parents) == keypoints.shape[1]:
            color_pink = 'pink'
            if j == 1 or j == 2:
                color_pink = 'black'

        col = 'red' if j in skeleton.joints_right() else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    #  plt.savefig('test/3Dimage_{}.png'.format(1000+num))
    width, height = fig.get_size_inches() * fig.get_dpi()
    _, t2 = ckpt_time(t1, desc='2 ')
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    cv2.imshow('im', image)
    cv2.waitKey(5)
    _, t3 = ckpt_time(t2, desc='3 ')
    return image
