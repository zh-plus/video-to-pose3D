# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import hashlib
import os
import pathlib
import shutil
import sys
import time

import cv2
import numpy as np
import torch


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def alpha_map(prediction):
    p_min, p_max = prediction.min(), prediction.max()

    k = 1.6 / (p_max - p_min)
    b = 0.8 - k * p_max

    prediction = k * prediction + b

    return prediction


def change_score(prediction, detectron_detection_path):
    detectron_predictions = np.load(detectron_detection_path, allow_pickle=True)['positions_2d'].item()
    pose = detectron_predictions['S1']['Directions 1']
    prediction[..., 2] = pose[..., 2]

    return prediction


class Timer:
    def __init__(self, message, show=True):
        self.message = message
        self.elapsed = 0
        self.show = show

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show:
            print(f'{self.message} --- elapsed time: {time.perf_counter() - self.start} s')


def calculate_area(data):
    """
    Get the rectangle area of keypoints.
    :param data: AlphaPose keypoint format, [x, y, score, ... , x, y, score]
    :return: area
    """
    data = np.reshape(data, (-1, 3))
    width = min(data[:, 0]) - max(data[:, 0])
    height = min(data[:, 1]) - max(data[:, 1])

    return np.abs(width * height)


def read_video(filename, fps=None, skip=0, limit=-1):
    stream = cv2.VideoCapture(filename)

    i = 0
    while True:
        grabbed, frame = stream.read()
        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            print('===========================> This video get ' + str(i) + ' frames in total.')
            sys.stdout.flush()
            break

        i += 1
        if i > skip:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield np.array(frame)
        if i == limit:
            break


def split_video(video_path):
    stream = cv2.VideoCapture(video_path)

    output_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_name = video_name[:video_name.rfind('.')]

    save_folder = pathlib.Path(f'{output_dir}/image/{video_name}')
    shutil.rmtree(str(save_folder), ignore_errors=True)
    save_folder.mkdir(parents=True, exist_ok=True)

    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    length = len(str(total_frames)) + 1

    i = 1
    while True:
        grabbed, frame = stream.read()

        if not grabbed:
            print(f'Split totally {i + 1} images from video.')
            break

        save_path = f'{output_dir}/image/{video_name}/output{str(i).zfill(length)}.png'
        cv2.imwrite(save_path, frame)

        i += 1

    return os.path.dirname(save_path)


if __name__ == '__main__':
    os.chdir('..')

    split_video('outputs/kobe.mp4')
