import os
import sys
import cv2
import ipdb
import argparse

import numpy as np

from tqdm import tqdm

from utils import *
from monovod import *
from plotting import *
from display import *

def main(video_path, poses_path=None, calibs_path=None, output_dir='html', focal_length=716):
    cap = cv2.VideoCapture(video_path)
    ret = True 

    monovod = MONOVOD(calibs_path, focal_length)
    disp_map = DisplayMap()
    cur_pose = np.eye(4)

    est_path = []

    if poses_path is not None:
        gt_path = [] 
        gt_poses = read_poses(poses_path)
        pbar = tqdm(total=len(gt_poses))
    else:
        gt_path = None
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) 

    while ret: 
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            f = monovod.add_frame(img)
            if f > 0: 
                q1, q2 = monovod.get_matches() 
                cur_pose = cur_pose @ monovod.get_pose(q1, q2)
                monovod.poses.append(cur_pose)
                disp_map.paint(monovod)
            
            if calibs_path is None and f == 0:
                monovod.fill_calib()

            pbar.update(1)
        
        if poses_path is not None: gt_path.append((gt_poses[f][0, 3], gt_poses[f][2, 3])) 
        est_path.append((cur_pose[0, 3], cur_pose[2, 3])) 
    video_name = os.path.basename(video_path).split('.')[0]
    visualize_paths(est_path, gt_path, f"MONOVOD_{video_name}", file_out= f"{output_dir}/{video_name}.html") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='path to the video you want to track')
    parser.add_argument('--poses_path', type=str, help='text file contatining the pose matrix for each frame in video')
    parser.add_argument('--calibs_path', type=str, help='text file contatining the projection matrix')
    parser.add_argument('--output_dir', type=str, default='html', help='folder to save html output files')
    parser.add_argument('--focal_length', type=int, default=716, help='focal length of the camera for building projection matrix if calibration path is not given')

    args = parser.parse_args()
    print(args)

    list_args = []
    for k in args.__dict__:
        list_args.append(args.__dict__[k])
    main(*list_args)
