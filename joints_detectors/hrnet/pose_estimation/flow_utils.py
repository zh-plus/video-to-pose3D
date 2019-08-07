import numpy as np
import torch

from flownet2.models import FlowNet2


def load_model():
    class parsers():
        # 'Run model in pseudo-fp16 mode (fp16 storage fp32 math)
        #  fp16 = True
        fp16 = False
        rgb_max = 255.0

    args = parsers()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/flow_net2/models/FlowNet2_checkpoint.pth.tar")
    #  dict = torch.load("/home/xyliu/2D_pose/deep-high-resolution-net.pytorch/flow_net/models/FlowNet2-S_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    net.eval()
    return net


def flow_net(img1, img2, net):
    # Prepare img pair
    # H x W x 3(RGB)
    im1 = img1[..., ::-1]
    im2 = img2[..., ::-1]
    images = [im1, im2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()

    # flow reslut for bbox propagation
    f_result = result.data.cpu().numpy()

    return f_result


def flow_propagation(keypoints, flow):
    """propagation from previous frame.
    Arguments:
        keypoints (ndarray): [num_people, num_keypoints, 3] (x, y, score)
                             keypoints detection of previous frame.
        flow (ndarray): [2, H, W]
                        optical flow between previous frame and current frame.
    Returns:
        boxes (ndarray): [num_people, 4]
                         boxes propagated from previous frame.
    """
    extend_factor = 0.15
    H = flow.shape[1]
    W = flow.shape[2]
    num_kpts = keypoints.shape[1]
    flow = flow.transpose((2, 1, 0))  # [W, H, 2]
    pos = keypoints[:, :, :2].reshape(-1, 2).T.astype(int)

    # pos 坐标要在H，W范围之内 
    pos[0] = pos[0] * (pos[0] < W)
    pos[1] = pos[1] * (pos[1] < H)
    pos = pos.tolist()

    offset = flow[tuple(pos)].reshape(-1, num_kpts, 2)
    shift_keypoints = keypoints[:, :, :2] + offset
    mask = keypoints[:, :, 2] > 0
    mask = mask[:, :, np.newaxis]
    # 选出shfit_keypoints的最大最小位置
    min_ = np.min(shift_keypoints + (1 - mask) * max(H, W), axis=1)  # [N,2]
    max_ = np.max(shift_keypoints * mask, axis=1)  # [N,2]
    extend = (max_ - min_) * extend_factor / 2
    up_left = np.fmax(min_ - extend, 0)
    bottom_right = np.fmin(max_ + extend, np.array([W - 1, H - 1]))
    boxes = np.concatenate((up_left, bottom_right), axis=1)

    return boxes, shift_keypoints
