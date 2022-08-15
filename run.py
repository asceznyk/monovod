import os
import sys
import cv2

import numpy as np

from tqdm import tqdm

from utils import *
from monovod import *
from plotting import *

def run(video_path, poses_path=None, calibs_path=None, fl=716):
    cap = cv2.VideoCapture(video_path)
    ret = True 

    monovod = MONOVOD(calibs_path, fl)
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
                tsfm = monovod.get_pose(q1, q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(tsfm))
            
            if calibs_path is None and f == 0:
                monovod.fill_calib()

            pbar.update(1)
        
        if poses_path is not None: gt_path.append((gt_poses[f][0, 3], gt_poses[f][2, 3])) 
        est_path.append((cur_pose[0, 3], cur_pose[2, 3])) 
    video_name = os.path.basename(video_path.replace('.avi', ''))
    visualize_paths(est_path, gt_path, f"MONOVOD_{video_name}", file_out= f"{video_name}.html") 

if __name__ == '__main__':
    print(sys.argv)
    run(*list(sys.argv[1:]))


