'''
先生成所有的2D坐标
再生成3D坐标，
再绘图，不是实时的
'''
import os
import cv2
from tqdm import tqdm
import time
import numpy as np
from argparse import ArgumentParser
import sys

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from pyqtgraph.opengl import *
from joints_detectors.openpose.main import load_model as Model2Dload
model2D = Model2Dload()
from joints_detectors.openpose.main import generate_frame_kpt as OpenPoseInterface
interface2D = OpenPoseInterface
from tools.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()
from tools.utils import interface as VideoPoseInterface
interface3D = VideoPoseInterface
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img, common
common = common()

# 先得到所有视频的2D坐标，再统一生成3D坐标
def VideoPoseJoints(VideoName):
    cap, cap_length = videoInfo(VideoName)
    kpt2Ds = []
    for i in tqdm(range(cap_length)):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            joint2D = interface2D(frame, model2D)
        except Exception as e:
            print(e)
            continue
        draw_2Dimg(frame, joint2D, 1)
        kpt2Ds.append(joint2D)

    joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
    return joint3D



item = 0
pos_init = np.zeros(shape=(17,3))
class Visualizer(object):
    def __init__(self, skeletons_3d):
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 45.0       ## distance of camera from center
        self.w.opts['fov'] = 60              ## horizontal field of view in degrees
        self.w.opts['elevation'] = 10       ## camera's angle of elevation in degrees 仰俯角
        self.w.opts['azimuth'] = 90         ## camera's azimuthal angle in degrees 方位角
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(450, 700, 980, 700) #原点在左上角
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)

        # special setting
        pos = pos_init
        self.skeleton_parents = common.skeleton_parents
        self.skeletons_3d = skeletons_3d


        for j, j_parent in enumerate(self.skeleton_parents):
            if j_parent == -1:
                continue
            x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
            y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
            z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10
            pos_total = np.vstack([x,y,z]).transpose()
            self.traces[j] = gl.GLLinePlotItem(pos=pos_total, color=pg.glColor((j, 10)), width=6,  antialias=True)
            self.w.addItem(self.traces[j])


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)


    def update(self):
        time.sleep(0.03)
        global item
        pos = self.skeletons_3d[item]
        print(item, '  ')
        item += 1

        for j, j_parent in enumerate(self.skeleton_parents):
            if j_parent == -1:
                continue

            x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
            y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
            z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10
            pos_total = np.vstack([x,y,z]).transpose()
            self.set_plotdata(
                name=j, points=pos_total,
                color=pg.glColor((j, 10)),
                width=6)


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(1)
        self.start()

def main(VideoName):
    print(VideoName)
    joint3D = VideoPoseJoints(VideoName)
    v = Visualizer(joint3D)
    v.animation()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-video", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    args = parser.parse_args()
    VideoName = args.video_input
    main(VideoName)
