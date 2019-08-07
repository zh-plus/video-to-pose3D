# Video to Pose3D
### Prerequisite

1. Environment
   - Linux system
   - Python 3+ distribution
2. Dependencies
   - Detectron, see [here](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)
   - Pytorch > 0.4.0
   - ffmpeg-python
   - Download weight file <model_final.pkl> from [here](https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl), and put it into data folder
   - Download checkpoint file <d-pt-243.bin> from [here](https://dl.fbaipublicfiles.com/video-pose-3d/d-pt-243.bin), and put it into checkpoint folder



### Quick start

1. place your video into `in_the_wild_data` folder. (I've prepared some videos).
2. change the `video_path` in the <vedio_pose.py>
3. Run it! You will find the rendered output video in the `in_the_wild_data` folder.



### Result

<p align="center"><img src="images/kunkun_cut_out.gif" width="120%" alt="" /></p>


### Advanced

As this script is based on the [VedioPose3D](https://github.com/facebookresearch/VideoPose3D) provided by Facebook, and automated in the following way:

```python
args = parse_args()

video_dir, video_name = os.path.dirname(video_path), os.path.basename(video_path)

args.keypoints = f'{keypoints_folder_path}/data_2d_detections.npz'
args.architecture = '3, 3, 3, 3, 3'
args.checkpoint = 'checkpoint'
args.evaluate = 'd-pt-243.bin'
args.render = True
args.viz_subject = 'S1'
args.viz_action = 'Directions'
args.viz_video = video_path
args.viz_camera = 0
args.viz_output = f'{video_dir}/{video_name[:video_name.rfind(".")]}_out.{"gif" if use_gif else "mp4"}'
args.viz_size = 5
args.viz_downsample = 1
# args.viz_skip = 9
args.fps = 25

print(args)

run(args)
```

The meaning of arguments can be found [here](https://github.com/facebookresearch/VideoPose3D/blob/master/DOCUMENTATION.md), you can customize it conveniently by changing the `args`.



### Acknowledgement

The 2D pose to 3D pose and visualization part is from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

Some of the "In the wild" script is adapted from the other [fork](https://github.com/tobiascz/VideoPose3D).



### Coming soon

The other feature will be added to improve accuracy in the future:

- [x] Human completeness check.
- [ ] Object Tracking to the first complete human covering largest area.
- [x] Change 2D pose estimation method such as [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [ ] Data augmentation to solve "high-speed with low-rate" problem: [SLOW-MO](https://github.com/avinashpaliwal/Super-SloMo)

