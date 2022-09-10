import cv2

import numpy as np

from utils import *

class MONOVOD():
    def __init__(self, calibs_path=None, f=716, n_feat=3000):
        self.orb = cv2.ORB_create(n_feat) 
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.frames = []
        self.f = f
        self.pm, self.cm = None, None
        if calibs_path is not None:
            self.pm, self.cm = read_calib(calibs_path)
            print(self.pm)

        self.poses = []

    def fill_calib(self):
        self.pm, self.cm = init_cam_intrinsics(self.frames[-1], self.f) 
        print(self.pm)

    def add_frame(self, f):
        self.frames.append(f)
        return len(self.frames)-1

    def get_matches(self, draw_matches=0):
        kp1, des1 = self.orb.detectAndCompute(self.frames[-2], None) 
        kp2, des2 = self.orb.detectAndCompute(self.frames[-1], None)

        matches = self.bf.knnMatch(des1, des2, k=2)

        good = []
        try:
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except:
            pass

        if draw_matches:
            draw_params = dict(matchColor = -1,
                               singlePointColor = None,
                               matchesMask = None,
                               flags = 2)

            img3 = cv2.drawMatches(self.frames[-2], kp1, self.frames[-1], kp2, good, None, **draw_params)
            cv2.imshow('doesnt_matter',img3)
            cv2.waitKey(200)

        return np.float64([kp1[m.queryIdx].pt for m in good]), np.float64([kp2[m.trainIdx].pt for m in good])
        
    def tsfm_mat(self, r, t):
        z = np.eye(4, dtype=np.float64)
        z[:3, :3] = r
        z[:3, 3] = t
        return z
    
    def calc_rt(self, e, q1, q2):
        def decompose_mat(e):
            w = np.array([[0,-1,0], [1,0,0], [0,0,1]])
            u, s, vt = np.linalg.svd(e)
            if np.linalg.det(u) < 0: u *= -1
            if np.linalg.det(vt) < 0: vt *= -1
            r1, r2 = u @ w @ vt, u @ w.T @ vt
            t = u[:,2]
            return r1, r2, t

        def sum_z_cal_relative_scale(r, t):
            x = self.tsfm_mat(r, t)
            hom_q1 = cv2.triangulatePoints(self.pm, self.pm @ x, q1.T, q2.T) 
            hom_q2 = x @ hom_q1

            uhom_q1 = hom_q1[:3, :] / (hom_q1[3, :] + 1e-24)
            uhom_q2 = hom_q2[:3, :] / (hom_q2[3, :] + 1e-24) 

            scale = np.mean(
                np.linalg.norm(uhom_q1.T[:-1] - uhom_q1.T[1:], axis=-1) / 
                np.linalg.norm(uhom_q2.T[:-1] - uhom_q2.T[1:], axis=-1),
                dtype=np.float128
            ) 
            return sum(uhom_q1[2, :] > 0) + sum(uhom_q2[2, :] > 0), scale 

        r1, r2, t = decompose_mat(e)
        pairs = [[r1, t], [r1, -t], [r2, t], [r2, -t]]
        sumzs = []
        for i, [r, t] in enumerate(pairs): 
            sumzs.append((i, *sum_z_cal_relative_scale(r, t)))

        j, _, s = max(sumzs, key=lambda x: x[1])
        r, t = pairs[j]
        return [r, s*t]

    def get_pose(self, q1, q2):
        e, _ = cv2.findEssentialMat(q2, q1, self.cm, threshold=1)
        [r, t] = self.calc_rt(e, q2, q1)
        return self.tsfm_mat(r, np.squeeze(t))

