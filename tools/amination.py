'''
Realtime Display 3D Human Reconstrction
3D image drawing by pygtagrph based on OpenGL
speed about 25 FPS
'''
import math
import sys
from argparse import ArgumentParser

import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.opengl import *

from joints_detectors.openpose.main import load_model as Model2Dload

model2D = Model2Dload()
from joints_detectors.openpose.main import generate_frame_kpt as OpenPoseInterface

interface2D = OpenPoseInterface
from tools.utils import videopose_model_load as Model3Dload

model3D = Model3Dload()
from tools.utils import interface as VideoPoseInterface

interface3D = VideoPoseInterface
from tools.utils import draw_2Dimg, resize_img, common

common = common()
item = 0
item_num = 0
pos_init = np.zeros(shape=(17, 3))


class Visualizer(object):
    def __init__(self, input_video):
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 45.0  ## distance of camera from center
        self.w.opts['fov'] = 60  ## horizontal field of view in degrees
        self.w.opts['elevation'] = 10  ## camera's angle of elevation in degrees 仰俯角
        self.w.opts['azimuth'] = 90  ## camera's azimuthal angle in degrees 方位角
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(450, 700, 980, 700)  # 原点在左上角
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
        self.cap = cv2.VideoCapture(input_video)
        self.video_name = input_video.split('/')[-1].split('.')[0]
        self.kpt2Ds = []
        pos = pos_init

        for j, j_parent in enumerate(common.skeleton_parents):
            if j_parent == -1:
                continue

            x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
            y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
            z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10
            pos_total = np.vstack([x, y, z]).transpose()
            self.traces[j] = gl.GLLinePlotItem(pos=pos_total, color=pg.glColor((j, 10)), width=6, antialias=True)
            self.w.addItem(self.traces[j])

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        global item
        global item_num
        num = item / 2
        azimuth_value = abs(num % 120 + math.pow(-1, int((num / 120))) * 120) % 120
        self.w.opts['azimuth'] = azimuth_value
        print(item, '  ')
        _, frame = self.cap.read()
        if item % 2 != 1:
            frame, W, H = resize_img(frame)
            joint2D = interface2D(frame, model2D)
            img2D = draw_2Dimg(frame, joint2D, 1)
            if item == 0:
                for _ in range(30):
                    self.kpt2Ds.append(joint2D)
            elif item < 30:
                self.kpt2Ds.append(joint2D)
                self.kpt2Ds.pop(0)
            else:
                self.kpt2Ds.append(joint2D)
                self.kpt2Ds.pop(0)

            item += 1
            joint3D = interface3D(model3D, np.array(self.kpt2Ds), W, H)
            pos = joint3D[-1]  # (17, 3)

            for j, j_parent in enumerate(common.skeleton_parents):
                if j_parent == -1:
                    continue
                x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
                y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
                z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10
                pos_total = np.vstack([x, y, z]).transpose()
                self.set_plotdata(
                    name=j, points=pos_total,
                    color=pg.glColor((j, 10)),
                    width=6)

            # save
            if item_num < 10:
                name = '000' + str(item_num)

            elif item_num < 100:
                name = '00' + str(item_num)

            elif item_num < 1000:
                name = '0' + str(item_num)

            else:
                name = str(item_num)
            im3Dname = 'VideoSave/' + '3D_' + name + '.png'
            d = self.w.renderToArray((img2D.shape[1], img2D.shape[0]))  # (W, H)
            print('Save 3D image: ', im3Dname)
            pg.makeQImage(d).save(im3Dname)

            im2Dname = 'VideoSave/' + '2D_' + name + '.png'
            print('Save 2D image: ', im2Dname)
            cv2.imwrite(im2Dname, img2D)

            item_num += 1
        else:
            item += 1

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(1)
        self.start()


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--video", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    args = parser.parse_args()
    print(args.video)
    v = Visualizer(args.video)
    v.animation()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
