import cv2

from tqdm import tqdm

path = '../outputs/nba2k.mp4'
stream = cv2.VideoCapture(path)
assert stream.isOpened(), 'Cannot capture source'

video_length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = stream.get(cv2.CAP_PROP_FPS)
video_size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), video_fps, video_size)

for i in tqdm(range(video_length)):
    i += 1
    grabbed, frame = stream.read()

    writer.write(frame)

    # if the `grabbed` boolean is `False`, then we have
    # reached the end of the video file
    if not grabbed:
        print('\n===========================> This video get ' + str(i) + ' frames in total.')
        break

writer.release()
