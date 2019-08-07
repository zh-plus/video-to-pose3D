'''
使用yolov3作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import cv2
from tqdm import tqdm

from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_final_preds
from lib.detector.yolo.human_detector import main as yolo_det
from pose_estimation.flow_utils import *
from pose_estimation.utilitys import plot_keypoint, preprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        #  default='experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml',
                        #  default='experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument("-i", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    parser.add_argument("-o", "--video_output", help="output video file name", default="output/output.mp4")

    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--display', action='store_true')

    args = parser.parse_args()
    return args


##### load model
def model_load(config):
    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False
    )
    model_file_name = 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
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


###### 加载human detecotor model
from lib.detector.yolo.human_detector import load_model as yolo_model

# sys.path.remove('/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/flow_net2')
human_model = yolo_model()


def main():
    args = parse_args()
    update_config(cfg, args)

    if not args.camera:
        # handle video
        cam = cv2.VideoCapture(args.video_input)
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cam = cv2.VideoCapture(0)
        video_length = 30000

    ret_val, input_image = cam.read()
    resize_W = 640
    resize_H = 384
    input_image = cv2.resize(input_image, (resize_W, resize_H))
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.video_output, fourcc, input_fps, (input_image.shape[1], input_image.shape[0]))

    #### load optical flow model
    flow_model = load_model()

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    first_frame = 1

    flow_boxs = 0
    flow_kpts = 0

    item = 0
    for i in tqdm(range(video_length - 1)):

        x0 = ckpt_time()
        ret_val, input_image = cam.read()
        input_image = cv2.resize(input_image, (resize_W, resize_H))

        if first_frame == 0:
            try:
                t0 = ckpt_time()
                flow_result = flow_net(pre_image, input_image, flow_model)
                flow_boxs, flow_kpts = flow_propagation(keypoints, flow_result)
                _, t1 = ckpt_time(t0, 1)
            except Exception as e:
                print(e)
                continue

        pre_image = input_image
        first_frame = 0

        try:
            bboxs, scores = yolo_det(input_image, human_model)
            # bbox is coordinate location
            if type(flow_boxs) == int:
                inputs, origin_img, center, scale = preprocess(input_image, bboxs, scores, cfg)
            else:
                #  flow_boxs = (flow_boxs + bboxs) /2
                inputs, origin_img, center, scale = preprocess(input_image, flow_boxs, scores, cfg)
        except:
            out.write(input_image)
            cv2.namedWindow("enhanced", 0);
            cv2.resizeWindow("enhanced", 1080, 720);
            cv2.imshow('enhanced', input_image)
            cv2.waitKey(2)
            continue

        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:, [2, 1, 0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        if type(flow_boxs) != int:
            preds = (preds + flow_kpts) / 2

        origin_img = np.zeros(origin_img.shape, np.uint8)
        image = plot_keypoint(origin_img, preds, maxvals, 0.1)
        out.write(image)
        keypoints = np.concatenate((preds, maxvals), 2)

        if args.display:
            ########### 指定屏幕大小
            cv2.namedWindow("enhanced", cv2.WINDOW_GUI_NORMAL);
            cv2.resizeWindow("enhanced", 1920, 1080);
            cv2.imshow('enhanced', image)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
