from tqdm import tqdm
import mediapipe as mp
import numpy as np
import cv2

pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def generate_kpts(video_file):
    vid = cv2.VideoCapture(video_file)
    kpts = []
    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(video_length)):
        ret, frame = vid.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if(not results.pose_landmarks):
            kpts.append(kpts[-1] if len(kpts) > 0 else [[0, 0] for _ in range(17)])
            continue
        
        # Take the coco keypoints
        l = results.pose_landmarks.landmark
        pl = mp.solutions.pose.PoseLandmark
        kpts.append([
            [l[pl.NOSE].x, l[pl.NOSE].y],
            [l[pl.LEFT_EYE].x, l[pl.LEFT_EYE].y],
            [l[pl.RIGHT_EYE].x, l[pl.RIGHT_EYE].y],
            [l[pl.LEFT_EAR].x, l[pl.LEFT_EAR].y],
            [l[pl.RIGHT_EAR].x, l[pl.RIGHT_EAR].y],
            [l[pl.LEFT_SHOULDER].x, l[pl.LEFT_SHOULDER].y],
            [l[pl.RIGHT_SHOULDER].x, l[pl.RIGHT_SHOULDER].y],
            [l[pl.LEFT_ELBOW].x, l[pl.LEFT_ELBOW].y],
            [l[pl.RIGHT_ELBOW].x, l[pl.RIGHT_ELBOW].y],
            [l[pl.LEFT_WRIST].x, l[pl.LEFT_WRIST].y],
            [l[pl.RIGHT_WRIST].x, l[pl.RIGHT_WRIST].y],
            [l[pl.LEFT_HIP].x, l[pl.LEFT_HIP].y],
            [l[pl.RIGHT_HIP].x, l[pl.RIGHT_HIP].y],
            [l[pl.LEFT_KNEE].x, l[pl.LEFT_KNEE].y],
            [l[pl.RIGHT_KNEE].x, l[pl.RIGHT_KNEE].y],
            [l[pl.LEFT_ANKLE].x, l[pl.LEFT_ANKLE].y],
            [l[pl.RIGHT_ANKLE].x, l[pl.RIGHT_ANKLE].y]
        ])

        # multiply all the x coordinates with frame width and y coordinates with frame height.
        for i in range(len(kpts[-1])):
            kpts[-1][i][0] *= frame.shape[1]
            kpts[-1][i][1] *= frame.shape[0]

    return np.array(kpts)