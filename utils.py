import numpy as np

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




