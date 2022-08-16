import cv2
import numpy as np

def init_cam_intrinsics(img, f):
    w, h = img.shape[1], img.shape[0]
    pm = np.array([
        [f, 0, w//2, 0],
        [0, f, h//2, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)    
    return pm, pm[:3, :3] 

def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            t = np.fromstring(line, dtype=np.float64, sep=' ')
            t = t.reshape(3, 4)
            t = np.vstack((t, [0, 0, 0, 1]))
            poses.append(t)
    return poses

def read_calib(file_path):
    with open(file_path, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        p = np.reshape(params, (3, 4))
    return p, p[:3, :3]

def triangulate_points(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        a = np.zeros((4,4))
        a[0] = p1[0] * pose1[2] - pose1[0] ##[-f, 0, p1x-w//2, 0]
        a[1] = p1[1] * pose1[2] - pose1[1] ##[0, -f, p1y-h//2, 0]
        a[2] = p2[0] * pose2[2] - pose2[0] ##[-f, 0, p2x-w//2, 0]
        a[3] = p2[1] * pose2[2] - pose2[1] ##[0, -f, p2y-h//2, 0]
        _, _, vt = np.linalg.svd(a)
        ret[i] = vt[3]
    return ret.T




