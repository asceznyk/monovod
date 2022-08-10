import os
import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

from utils import *
from vodom import *
from plotting import *

def init_image_scale(img, scl=2):
    h, w = img.shape[1]//scl, img.shape[0]//scl
    f = 716//scl ##these are pre-selected values 
    cm = np.array([
        [f, 0, h//2],
        [0, f, w//2],
        [0, 0, 1],
    ])

    return h, w, cm, (f, f)

def run(video_path, poses_path, calibs_path):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    _, img = cap.read()
    h, w, cm, _ = init_image_scale(img, scl=1)
    vodom = vODOM(calibs_path, None)
    cur_pose = np.eye(4)

    est_path = []
    gt_path = [] 
    gt_poses = read_poses(poses_path)
    while ret: 
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (h,w)) 
            if f > 1: 
                q1, q2 = vodom.get_matches(draw_matches=0)
                tsfm = vodom.get_pose(q1, q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(tsfm))
            f = vodom.add_frame(img)

        gt_path.append((gt_poses[f][0, 3], gt_poses[f][2, 3]))
        est_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    
    visualize_paths(gt_path, est_path, f"vSLAM_{video_path}", file_out=os.path.basename(video_path.replace('.avi', '')) + ".html")
    return

run(sys.argv[1], sys.argv[2], sys.argv[3])

