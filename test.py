import os
import sys
import cv2

import numpy as np

from tqdm import tqdm

from utils import *
from vodom import *
from plotting import *

def run(video_path, poses_path, calibs_path, scl=2):
    cap = cv2.VideoCapture(video_path)
    ret = True    

    vodom = vODOM(calibs_path, None)
    cur_pose = np.eye(4)

    est_path = []
    gt_path = [] 
    gt_poses = read_poses(poses_path)
    pbar = tqdm(total=len(gt_poses))
    while ret: 
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            f = vodom.add_frame(img)

            if f > 0: 
                q1, q2 = vodom.get_matches()
                tsfm = vodom.get_pose(q1, q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(tsfm))  

            pbar.update(1)

        gt_path.append((gt_poses[f][0, 3], gt_poses[f][2, 3]))
        est_path.append((cur_pose[0, 3], cur_pose[2, 3])) 
    visualize_paths(gt_path, est_path, "vODOM: visual ODOMetry", file_out=os.path.basename(video_path.replace('.avi', '')) + ".html") 

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])

