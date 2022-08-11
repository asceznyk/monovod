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




