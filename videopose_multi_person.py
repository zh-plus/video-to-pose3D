from joints_detectors.Alphapose.gene_npz import handle_video
from pose_trackers.PoseFlow.tracker_general import track

if __name__ == '__main__':
    video = 'kobe.mp4'
    handle_video(f'outputs/{video}')
    track(video)

    # TODO: Need more action:
    #  1. "Improve the accuracy of tracking algorithm" or "Doing specific post processing after getting the track result".
    #  2. Choosing person(remove the other 2d points for each frame)
