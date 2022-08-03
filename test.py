import sys
import cv2

import numpy as np

import matlplotlib.pylot as plt

orb = cv2.ORB_create()

def get_orb_for_frames(video_path, orb, max_len=1):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0
    while ret and f < max_len: 
        ret, img = cap.read() 
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kps = orb.detect(img)
            kps, des = orb.compute(img, kps)
            kpimg = cv2.drawKeyPoints(img, kps, None, color=(0, 255, 0), flags=0)
            
            plt.imshow(kpimg)
            plt.show()
            plt.savefig(kpimg, 'keyframe.jpg')

        f += 1

    return

get_orb_for_frames('videos/test_countryroad.mp4', max_len=1)
