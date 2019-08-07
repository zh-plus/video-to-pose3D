import numpy as np


def soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.3, method=0):
    N = boxes.size(0)
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    for i in range(N):
        maxscore = scores[i]
        maxpos = i
        tx1 = boxes[i, 0]  # temp
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = scores[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < scores[pos]:
                maxscore = scores[pos]
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        scores[i] = scores[maxpos]

        # swap
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        scores[maxpos] = ts

        tx1 = boxes[i, 0]  # temp
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = scores[i]
        keep[count] = i
        pos = i + 1

        # NMS iterations
        while pos < N:
            x1 = boxes[i, 0]  # temp
            y1 = boxes[i, 1]
            x2 = boxes[i, 2]
            y2 = boxes[i, 3]
            s = scores[i]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(ty1, y1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, x1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # IoU

                    #  import ipdb;ipdb.set_trace()
                    if method == 1:  # LINEAR
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1

                    elif method == 2:  # Gaussain
                        weight = np.exp(-(ov * ov) / sigma).data.numpy()
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    scores[pos] = weight * scores[pos]

                    # discard the score less than threshold
                    # update N
                    if scores[pos] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        scores[pos] = scores[N - 1]
                        N = N - 1
                        pos = pos - 1
            pos = pos + 1

    keep = [i for i in range(N)]
    return keep, N
