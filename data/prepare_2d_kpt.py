import argparse
import os
import zipfile
import numpy as np
import h5py
import re
from glob import glob
from shutil import rmtree
from data_utils import suggest_metadata, suggest_pose_importer
import ipdb

import sys
sys.path.append('../')
from common.utils import wrap
from itertools import groupby

output_prefix_2d = 'data_2d_h36m_'
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description='Human3.6M dataset converter')

    parser.add_argument('-i', '--input', default='', type=str, metavar='PATH', help='input path to 2D detections')
    parser.add_argument('-o', '--output', default='detectron_pt_coco', type=str, metavar='PATH', help='output suffix for 2D detections (e.g. detectron_pt_coco)')

    args = parser.parse_args()

    if not args.input:
        print('Please specify the input directory')
        exit(0)


    # according to output name,generate some format. we use detectron
    import_func = suggest_pose_importer('detectron_pt_coco')
    metadata = suggest_metadata('detectron_pt_coco')

    print('Parsing 2D detections from', args.input)
    keypoints = import_func(args.input)

    output=keypoints.astype(np.float32)
    # 生成的数据用于后面的3D检测
    np.savez_compressed(output_prefix_2d + 'test' + args.output, positions_2d=output, metadata=metadata)
    print('npz name is ', output_prefix_2d + 'test' + args.output)

