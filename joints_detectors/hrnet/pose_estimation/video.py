'''
使用yolov3作为pose net模型的前处理
use yolov3 as the 2d human bbox detector
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_final_preds
from lib.detector.yolo.human_detector import load_model as yolo_model
from lib.detector.yolo.human_detector import main as yolo_det
from lib.models.pose_hrnet import get_pose_net as get_hr_pose_net
from lib.models.pose_resnet import get_pose_net as get_res_pose_net
from lib.utils.transforms import *
from pose_estimation.utilitys import preprocess

# path1 = os.path.split(os.path.realpath(__file__))[0]
# path2 = os.path.join(path1, '..')
# sys.path.insert(0, path1)
# sys.path.insert(0, path2)

# sys.path.pop(0)
# sys.path.pop(1)
# sys.path.pop(2)

kpt_queue = []


def smooth_filter(kpts):
    if len(kpt_queue) < 6:
        kpt_queue.append(kpts)
        return kpts

    queue_length = len(kpt_queue)
    if queue_length == 50:
        kpt_queue.pop(0)
    kpt_queue.append(kpts)

    # transpose to shape (17, 2, num, 50) 关节点keypoints num、横纵坐标、每帧人数、帧数
    transKpts = np.array(kpt_queue).transpose(1, 2, 3, 0)

    window_length = queue_length - 1 if queue_length % 2 == 0 else queue_length - 2
    # array, window_length(bigger is better), polyorder
    result = savgol_filter(transKpts, window_length, 3).transpose(3, 0, 1, 2)  # shape(frame_num, human_num, 17, 2)

    # 返回倒数第几帧 return third from last frame
    return result[-3]


class get_args():
    # hrnet config
    cfg = 'joints_detectors/hrnet/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
    dataDir = ''
    logDir = ''
    modelDir = ''
    opts = []
    prevModelDir = ''


def model_load(config):
    models_map = {
        'pose_resnet': get_res_pose_net,
        'pose_hrnet': get_hr_pose_net
    }

    model = models_map[config.MODEL.NAME](config, is_train=False)

    # model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
    #     config, is_train=False
    # )
    model_file_name = 'joints_detectors/hrnet/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def ckpt_time(t0=None, display=None):
    if not t0:
        return time.time()
    else:
        t1 = time.time()
        if display:
            print('consume {:2f} second'.format(t1 - t0))
        return t1 - t0, t1


def generate_kpts(video_name, smooth=False):
    human_model = yolo_model()
    args = get_args()
    update_config(cfg, args)
    cam = cv2.VideoCapture(video_name)
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # # ret_val, input_image = cam.read()
    # # Video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # input_fps = cam.get(cv2.CAP_PROP_FPS)

    pose_model = model_load(cfg)
    pose_model.cuda()

    # collect keypoints coordinate
    kpts_result = []
    for i in tqdm(range(video_length)):

        ret_val, input_image = cam.read()

        try:
            bboxs, scores = yolo_det(input_image, human_model)
            # bbox is coordinate location
            inputs, origin_img, center, scale = preprocess(input_image, bboxs, scores, cfg)
        except Exception as e:
            print(e)
            continue

        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:, [2, 1, 0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

            # if len(preds) != 1:
            #     print('here')

        if smooth:
            # smooth and fine-tune coordinates
            preds = smooth_filter(preds)

        # 3D video pose (only support single human)
        kpts_result.append(preds[0])

    result = np.array(kpts_result)
    return result


def getTwoModel():
    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    # load YoloV3 Model
    bbox_model = yolo_model()

    return bbox_model, pose_model


def getKptsFromImage(human_model, pose_model, image, smooth=None):
    args = get_args()
    update_config(cfg, args)

    bboxs, scores = yolo_det(image, human_model)
    # bbox is coordinate location
    inputs, origin_img, center, scale = preprocess(image, bboxs, scores, cfg)

    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:, [2, 1, 0]]
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # 3D video pose (only support single human)
    return preds[0]


def main():
    pass


if __name__ == '__main__':
    main()
