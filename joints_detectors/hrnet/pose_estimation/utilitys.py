import torch
import torchvision.transforms as transforms

from lib.utils.transforms import *

joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
               [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
               [5, 11], [6, 12], [11, 12],
               [11, 13], [12, 14], [13, 15], [14, 16]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255]]


def plot_keypoint(image, coordinates, confidence, keypoint_thresh):
    # USE cv2
    joint_visible = confidence[:, :, 0] > keypoint_thresh

    for i in range(coordinates.shape[0]):
        pts = coordinates[i]
        for color_i, jp in zip(colors, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt0 = pts[jp, 0];
                pt1 = pts[jp, 1]
                pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])

                cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 6)
                #  cv2.circle(image,(pt0_0, pt0_1), 2, color_i, thickness=-1)
                #  cv2.circle(image,(pt1_0, pt1_1), 2, color_i, thickness=-1)
    return image


def upscale_bbox_fn(bbox, img, scale=1.25):
    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(256, 192), scale=1.25):
    L = class_IDs.shape[1]
    thr = 0.5
    upscale_bbox = []
    for i in range(L):
        if class_IDs[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.asnumpy().tolist(), img, scale=scale))
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape)
        pose_input = pose_input.as_in_context(ctx)
    else:
        pose_input = None
    return pose_input, upscale_bbox


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def preprocess(image, bboxs, scores, cfg, thred_score=0.8):
    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = image

    inputs = []
    centers = []
    scales = []

    score_num = np.sum(scores > thred_score)
    max_box = min(5, score_num)
    for bbox in bboxs[:max_box]:
        x1, y1, x2, y2 = bbox
        box = [x1, y1, x2 - x1, y2 - y1]

        # 截取 box fron image  --> return center, scale
        c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input = transform(input).unsqueeze(0)
        inputs.append(input)

    inputs = torch.cat(inputs)
    return inputs, data_numpy, centers, scales
