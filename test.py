import sys
import cv2

import numpy as np

import matplotlib.pyplot as plt

orb = cv2.ORB_create(nfeatures=100000)

def orb_keyframe(img):
    gx, gy = 10, 10
    w,h = img.shape[0] // 2, img.shape[1] // 2
    img = cv2.resize(img, (h,w)) 

    kps, des = [], []
    sy = h // gy
    sx = w // gx

    print(h, w, sy, sx)

    for ry in range(0, h, sy):
        for rx in range(0, w, sx):
            _kp, _des = orb.compute(img[rx:rx+sx, ry:ry+sy], orb.detect(img))
            kps.extend(_kp)
            #cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1, 1)
            #des.extend(_des)

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

