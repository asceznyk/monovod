import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

from detector import *

def vslam(video_path, max_len=1):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    while ret and f < max_len: 
        ret, img = cap.read()
        if ret:
            w,h = img.shape[0] // 2, img.shape[1] // 2 
            img = cv2.resize(img, (h,w)) 
            kps, des = gftt_keyframe(img)  
            cv2.imwrite(f"keyframe.jpg", kpimg)

        f += 1

    return

vslam('/content/videos/test_countryroad.mp4', max_len=1)

