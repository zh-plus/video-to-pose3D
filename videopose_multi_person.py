from common.utils import split_video, add_path
from joints_detectors.Alphapose.gene_npz import generate_kpts
from pose_trackers.PoseFlow.tracker_general import track

add_path()

if __name__ == '__main__':
    video = 'outputs/nba2k.mp4'

    print(f'Splitting video: {video} into images...')
    split_video(video)

    generate_kpts(video)
    track(video)

    # TODO: Need more action:
    #  1. "Improve the accuracy of tracking algorithm" or "Doing specific post processing after getting the track result".
    #  2. Choosing person(remove the other 2d points for each frame)
