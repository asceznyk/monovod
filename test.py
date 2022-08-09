import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

from vslam import *

def run(video_path):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    vslam = vSLAM()
    while ret: 
        ret, img = cap.read()
        if ret:
            w,h = img.shape[0] // 2, img.shape[1] // 2 
            img = cv2.resize(img, (h,w)) 
            if f > 1: 
                q1, q2 = vslam.get_matches() 
            f = vslam.add_frame(img)
    return

run(sys.argv[1])

