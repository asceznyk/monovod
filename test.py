import os
import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

from plotting import *
from vslam import *

def init_image_scale(img, scl=2):
    h, w = img.shape[1]//scl, img.shape[0]//scl
    fx, fy = 716//scl, 716//scl ##these are pre-selected values 
    cm = np.array([
        [fx, 0, w//2],
        [0, fy, h//2],
        [0, 0, 1],
    ])

    return h, w, cm, (fx, fy)

def read_gt(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            t = np.fromstring(line, dtype=np.float64, sep=' ')
            t = t.reshape(3, 4)
            t = np.vstack((t, [0, 0, 0, 1]))
            poses.append(t)
    return poses

def run(video_path, txt_path):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    _, img = cap.read()
    h, w, cm, _ = init_image_scale(img, scl=1)
    vslam = vSLAM(cm)
    cur_pose = np.eye(4)

    est_path = []
    gt_path = [] 
    print(gt_path)
    gt_poses = read_gt(txt_path)
    while ret: 
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (h,w)) 
            if f > 1: 
                q1, q2 = vslam.get_matches()
                tsfm = vslam.get_pose(q1,q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(tsfm))
            f = vslam.add_frame(img)

        gt_path.append((gt_poses[f][0, 3], gt_poses[f][2, 3]))
        est_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    
    visualize_paths(gt_path, est_path, f"vSLAM_{video_path}", file_out=os.path.basename(video_path.replace('.avi', '')) + ".html")
    return

run(sys.argv[1], sys.argv[2])

