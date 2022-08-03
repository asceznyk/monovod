import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

orb = cv2.ORB_create(100)

def orb_keyframe(img):
    w,h = img.shape[0] // 2, img.shape[1] // 2
    img = cv2.resize(img, (h,w)) 
    kps, des = orb.compute(img, orb.detect(img))
    return cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)

def vslam(video_path, orb, max_len=1):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0
    while ret and f < max_len: 
        ret, img = cap.read()
        if ret:
            kpimg = orb_keyframe(img)
            plt.imshow(kpimg)
            plt.show()
            cv2.imwrite(f"keyframe.jpg", kpimg)

        f += 1

    return

vslam('/content/videos/test_countryroad.mp4', orb, max_len=1)

