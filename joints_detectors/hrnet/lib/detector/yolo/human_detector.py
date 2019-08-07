from __future__ import division

import os
import sys
import time

# scipt dirctory
yolo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, yolo_dir)

from lib.detector.yolo.util import *
import os.path as osp
from lib.detector.yolo.darknet import Darknet
from lib.detector.yolo.preprocess import prep_image

# sys.path.pop(0)

num_classes = 80


def ckpt_time(t0=None, display=1):
    if not t0:
        return time.time()
    else:
        ckpt = time.time() - t0
        if display:
            print('consume time {:3f}s'.format(ckpt))
        return ckpt, time.time()


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


class args():
    bs = 1
    nms_thresh = 0.4
    cfgfile = yolo_dir + '/cfg/yolov3.cfg'
    weightsfile = yolo_dir + '/yolov3.weights'
    reso = '416'
    scales = '1,2,3'
    confidence = 0.5


def load_model():
    scales = args.scales
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    classes = load_classes(yolo_dir + '/data/coco.names')
    # Set up the neural network
    print("Loading YOLO network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model


def main(images, model=None):
    '''images是path or image 矩阵   返回的是human bbox'''
    #  t0 = ckpt_time()

    scales = args.scales
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    classes = load_classes(yolo_dir + '/data/coco.names')

    if not model:
        # Set up the neural network
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()

    read_dir = time.time()
    # Detection phase
    if type(images) == str:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    else:
        imlist = []
        imlist.append(images)

    load_batch = time.time()

    inp_dim = int(model.net_info["height"])
    #  import ipdb;ipdb.set_trace()
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))])) for i in range(num_batches)]

    i = 0

    write = False
    start_det_loop = time.time()

    objs = {}

    for batch in im_batches:
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # Apply offsets to the result predictions
        # Tranform the predictions as described in the YOLO paper
        # flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        # Put every proposed box as a row.
        #  import ipdb;ipdb.set_trace()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        #        prediction = prediction[:,scale_indices]

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        end = time.time()

        #        print(end - start)

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    #  ckpt, t3 = ckpt_time(t2)
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    #################################
    # select human and export bbox
    #################################
    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]
        im_id = item[-1]
        if int(im_id) == 0:
            bbox = item[1:5].cpu().numpy()
            # conver float32 to .2f data
            bbox = [round(i, 2) for i in list(bbox)]
            score = item[5]
            bboxs.append(bbox)
            scores.append(score)
    scores = np.expand_dims(np.array(scores), 0)
    bboxs = np.array(bboxs)

    return bboxs, scores
