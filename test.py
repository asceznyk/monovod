import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

orb = cv2.ORB_create(nfeatures=100000)

def orb_keyframe(img):
    w,h = img.shape[0] // 2, img.shape[1] // 2 
    kps, des = orb.compute(img, orb.detect(img))
    return cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

def gftt_keyframe(img):
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

def vslam(video_path, orb, max_len=1):
    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    while ret and f < max_len: 
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (h,w)) 
            #kpimg = orb_keyframe(img)
            kpimg = gftt_keyframe(img)
            plt.imshow(kpimg)
            plt.show()
            cv2.imwrite(f"keyframe.jpg", kpimg)

        f += 1

    return

vslam('/content/videos/test_countryroad.mp4', orb, max_len=1)

