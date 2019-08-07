import os
import cv2
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
#
# from joints_detectors.openpose.main import load_model as Model2Dload
#
# model2D = Model2Dload()
# from joints_detectors.openpose.main import generate_frame_kpt as OpenPoseInterface
#
# interface2D = OpenPoseInterface
# from tools.utils import videopose_model_load as Model3Dload
#
# model3D = Model3Dload()
# from tools.utils import interface as VideoPoseInterface
#
# interface3D = VideoPoseInterface
# from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img
#
#
# def main(VideoName):
#     cap, cap_length = videoInfo(VideoName)
#     kpt2Ds = []
#     for i in tqdm(range(cap_length)):
#         _, frame = cap.read()
#         frame, W, H = resize_img(frame)
#
#         try:
#             joint2D = interface2D(frame, model2D)
#         except Exception as e:
#             print(e)
#             continue
#
#         if i == 0:
#             for _ in range(30):
#                 kpt2Ds.append(joint2D)
#         elif i < 30:
#             kpt2Ds.append(joint2D)
#             kpt2Ds.pop(0)
#         else:
#             kpt2Ds.append(joint2D)
#
#         joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
#         joint3D_item = joint3D[-1]  # (17, 3)
#         draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)
#
#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("-video", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
#     args = parser.parse_args()
#     VideoName = args.video_input
#     print('Input Video Name is ', VideoName)
#     main(VideoName)
