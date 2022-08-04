import cv2
import numpy as np

def gftt_keyframe(img):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return kps, des
    #return cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)


