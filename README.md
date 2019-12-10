# Video to Pose3D

> Predict 3d human pose from video

<p align="center"><img src="asset/kunkun_alphapose.gif" width="100%" alt="" /></p>

## Prerequisite

1. Environment
   - Linux system
   - Python > 3.6 distribution
2. Dependencies
   - **Packages**
      - Pytorch > 1.0.0
      - [torchsample](https://github.com/MVIG-SJTU/AlphaPose/issues/71#issuecomment-398616495)
      - [ffmpeg](https://ffmpeg.org/download.html)
      - tqdm
      - pillow
      - scipy
      - pandas
      - h5py
      - visdom
      - nibabel
      - opencv-python (install with pip)
      - matplotlib
   - **2D Joint detectors**
     - Alphapose (Recommended)
       - Download **duc_se.pth** from ([Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)),
         place to `./joints_detectors/Alphapose/models/sppe`
       - Download **yolov3-spp.weights** from ([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)),
         place to `./joints_detectors/Alphapose/models/yolo`
     - HR-Net (Bad 3d joints performance in my testing environment)
       - Download **pose_hrnet*** from [Google Drive](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) | [Baidu pan](https://pan.baidu.com/s/1v6dov-TyPVOtejTNF1HXrA)), 
         place to `./joints_detectors/hrnet/models/pytorch/pose_coco/`
       - Download **yolov3.weights** from [here](https://pjreddie.com/media/files/yolov3.weights),
         place to `./joints_detectors/hrnet/lib/detector/yolo`
     - OpenPose (Not tested, PR to README.md is highly appreciated )
   - **3D Joint detectors**
      - Download **pretrained_h36m_detectron_coco.bin** from [here](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin),
        place it into `./checkpoint` folder
   - **2D Pose trackers (Optional)**
      - PoseFlow (Recommended)
        No extra dependences
      - LightTrack (Bad 2d tracking performance in my testing environment)
        - See [original README](https://github.com/Guanghan/lighttrack), and perform same *get started step* on `./pose_trackers/lighttrack`



## Usage

0. place your video into `./outputs` folder. (I've prepared a test video).

##### Single person video

1. change the `video_path` in the `./videopose.py`
2. Run it! You will find the rendered output video in the `./outputs` folder.

##### Multiple person video (Not implemented yet)

1. For developing, check `./videopose_multi_person`

   ```python
   video = 'kobe.mp4'
   
   handle_video(f'outputs/{video}') 
   # Run AlphaPose, save the result into ./outputs/alpha_pose_kobe
   
   track(video)					 
   # Taking the result from above as the input of PoseTrack, output poseflow-results.json # into the same directory of above. 
   # The visualization result is save in ./outputs/alpha_pose_kobe/poseflow-vis
   
   # TODO: Need more action:
   #  1. "Improve the accuracy of tracking algorithm" or "Doing specific post processing 
   #     after getting the track result".
   #  2. Choosing person(remove the other 2d points for each frame)
   ```




##### Tips
0. The [PyCharm](https://www.jetbrains.com/pycharm/) is recommended since it is the IDE I'm using during development.
1. If you're using PyCharm, mark `./joints_detectors/Alphapose`, `./joints_detectors/hrnet` and `./pose_trackers` as source root.
2. If your're trying to run in command line, add these paths mentioned above to the sys.path at the head of `./videopose.py`

## Advanced

As this script is based on the [VedioPose3D](https://github.com/facebookresearch/VideoPose3D) provided by Facebook, and automated in the following way:

```python
args = parse_args()

args.detector_2d = 'alpha_pose'
dir_name = os.path.dirname(video_path)
basename = os.path.basename(video_path)
video_name = basename[:basename.rfind('.')]
args.viz_video = video_path
args.viz_output = f'{dir_name}/{args.detector_2d}_{video_name}.gif'

args.evaluate = 'pretrained_h36m_detectron_coco.bin'

with Timer(video_path):
    main(args)
```

The meaning of arguments can be found [here](https://github.com/facebookresearch/VideoPose3D/blob/master/DOCUMENTATION.md), you can customize it conveniently by changing the `args` in `./videopose.py`.



## Acknowledgement

The 2D pose to 3D pose and visualization part is from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

Some of the "In the wild" script is adapted from the other [fork](https://github.com/tobiascz/VideoPose3D).

The project structure and `./videopose.py` running script is adapted from [this repo](https://github.com/lxy5513/videopose)



## Coming soon

The other feature will be added to improve accuracy in the future:

- [x] Human completeness check.
- [x] Object Tracking to the first complete human covering largest area.
- [x] Change 2D pose estimation method such as [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).
- [x] Test HR-Net as 2d joints detector.
- [x] Test LightTrack as pose tracker.
- [ ] Multi-person video(complex) support.
- [ ] Data augmentation to solve "high-speed with low-rate" problem: [SLOW-MO](https://github.com/avinashpaliwal/Super-SloMo).

