from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from joints_detectors.Alphapose.gene_npz import handle_video


def remove_no_person_frames(video_path):
    results, name = handle_video(video_path)
    head, tail = 0, len(results)
    for result in results:
        if result['result']:
            break
        head += 1

    for result in reversed(results):
        if result['result']:
            break
        head += 1

    ffmpeg_extract_subclip(video_path, head, tail, video_path)


if __name__ == '__main__':
    remove_no_person_frames('../outputs/gait_test/001-bg-01-090.avi')
