import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


# visualization
def display_pose(imgdir, visdir, tracked, cmap):
    """
    Display 2d pose with person id in images.
    :param imgdir: split images from video.
    :param visdir: track visualization images saved here.
    :param tracked: track information for images.
    :param cmap: colormap instances for displaying distinct person.
    :return:
    """

    print("Start visualization...\n")
    pbar = tqdm(total=len(tracked))

    pool = Pool()
    for t in tracked.items():
        pool.apply_async(viz_one_pose, args=(t, imgdir, visdir, cmap), callback=lambda x: pbar.update())
    pool.close()
    pool.join()

    pbar.close()


def viz_one_pose(tracked_item, imgdir, visdir, cmap):
    imgname, content = tracked_item

    img = Image.open(os.path.join(imgdir, imgname))
    width, height = img.size
    fig = plt.figure(figsize=(width / 10, height / 10), dpi=10)
    plt.imshow(img)
    for pid in range(len(content)):
        pose = np.array(content[pid]['keypoints']).reshape(-1, 3)[:, :3]
        tracked_id = content[pid]['idx']

        # keypoint scores of torch version and pytorch version are different
        if np.mean(pose[:, 2]) < 1:
            alpha_ratio = 1.0
        else:
            alpha_ratio = 5.0

        if pose.shape[0] == 16:
            mpii_part_names = ['RAnkle', 'RKnee', 'RHip', 'LHip', 'LKnee', 'LAnkle', 'Pelv', 'Thrx', 'Neck', 'Head', 'RWrist', 'RElbow',
                               'RShoulder', 'LShoulder', 'LElbow', 'LWrist']
            colors = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'm', 'r', 'r', 'b', 'b']
            pairs = [[8, 9], [11, 12], [11, 10], [2, 1], [1, 0], [13, 14], [14, 15], [3, 4], [4, 5], [8, 7], [7, 6], [6, 2], [6, 3], [8, 12],
                     [8, 13]]
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c, 0], 0, width), np.clip(pose[idx_c, 1], 0, height), marker='o',
                         color=color, ms=80 / alpha_ratio * np.mean(pose[idx_c, 2]),
                         markerfacecolor=(1, 1, 0, 0.7 / alpha_ratio * pose[idx_c, 2]))
            for idx in range(len(pairs)):
                plt.plot(np.clip(pose[pairs[idx], 0], 0, width), np.clip(pose[pairs[idx], 1], 0, height), 'r-',
                         color=cmap(tracked_id), linewidth=60 / alpha_ratio * np.mean(pose[pairs[idx], 2]),
                         alpha=0.6 / alpha_ratio * np.mean(pose[pairs[idx], 2]))
        elif pose.shape[0] == 17:
            coco_part_names = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip',
                               'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
            colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g', 'g', 'g', 'g']
            pairs = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
                     [6, 12], [5, 11]]
            for idx_c, color in enumerate(colors):
                plt.plot(np.clip(pose[idx_c, 0], 0, width), np.clip(pose[idx_c, 1], 0, height), marker='o',
                         color=color, ms=80 / alpha_ratio * np.mean(pose[idx_c, 2]),
                         markerfacecolor=(1, 1, 0, 0.7 / alpha_ratio * pose[idx_c, 2]))

            for idx in range(len(pairs)):
                plt.plot(np.clip(pose[pairs[idx], 0], 0, width), np.clip(pose[pairs[idx], 1], 0, height), 'r-',
                         color=cmap(tracked_id), linewidth=60 / alpha_ratio * np.mean(pose[pairs[idx], 2]),
                         alpha=0.6 / alpha_ratio * np.mean(pose[pairs[idx], 2]))

            middle_ankle = (pose[-1] + pose[-2]) / 2
            # plt.annotate(str(tracked_id), xy=tuple(middle_ankle[:2]))
            plt.text(middle_ankle[0], middle_ankle[1], str(tracked_id), fontsize=60)

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if not os.path.exists(visdir):
        os.mkdir(visdir)
    fig.savefig(os.path.join(visdir, imgname[: imgname.rfind('.')] + ".png"), pad_inches=0.0, bbox_inches=extent, dpi=13)
    plt.close()
