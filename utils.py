import numpy as np

def init_image_scale(img, scl=2):
    h, w = img.shape[1]//scl, img.shape[0]//scl
    f = 716//scl ##these are pre-selected values 
    cm = np.array([
        [f, 0, h//2],
        [0, f, w//2],
        [0, 0, 1],
    ])

    return h, w, cm, (f, f)

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




