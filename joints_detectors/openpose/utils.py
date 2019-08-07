import cv2


joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 85, 255]]


def plot_keypoint(image, keypoints, keypoint_thresh=0.1):
    confidence = keypoints[:,:,2:]
    coordinates = keypoints[:,:,:2]
    joint_visible = confidence[:, :, 0] > keypoint_thresh


    # 描点
    for people in keypoints:
        for i in range(len(people)):
            x, y, p = people[i]
            if p < 0.1:
                continue
            x = int(x)
            y = int(y)
            cv2.circle(image, (x, y), 4, colors[i], thickness=-1)

    for i in range(coordinates.shape[0]):
        pts = coordinates[i]
        for color_i, jp in zip(colors, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt0 = pts[jp, 0];pt1 = pts[jp, 1]
                pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])
                cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 2)
    return image


# convert openpose keypoints(25) format to coco keypoints(17) format
def convert(op_kpts):
    '''
    0-16 map to 0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11
    '''
    coco_kpts = []
    for i, j in enumerate([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]):
        score = op_kpts[j][-1]
        # if eye, ear keypoints score is lower, map it to mouth
        if score < 0.2 and j in [15, 16, 17, 18]:
            coco_kpts.append(op_kpts[0])
        else:
            coco_kpts.append(op_kpts[j])

    return coco_kpts


# convert openpose keypoints(25) format to keypoints(18) format
def convert_18(op_kpts):
    coco_kpts = []
    for i, j in enumerate(range(0, 18)):
        if i<8:
            coco_kpts.append(op_kpts[j])
        else:
            coco_kpts.append(op_kpts[j+1])
    return coco_kpts
