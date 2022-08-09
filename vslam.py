import cv2
import numpy as np

from utils import *

class vSLAM():
    def __init__(self, calibs_path, cm=None):
        self.orb = cv2.ORB_create(3000) 
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.frames = []
        self.pm, _cm = read_calib(calibs_path)
        self.cm = cm if cm is not None else _cm
        print(self.cm)

    def add_frame(self, f):
        self.frames.append(f)
        return len(self.frames)

    def get_matches(self, draw_matches=1):
        img1, img2 = self.frames[-2], self.frames[-1]
        kp1, des1 = self.orb.detectAndCompute(img1, None) 
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if draw_matches:
            draw_params = dict(matchColor = -1,
                               singlePointColor = None,
                               matchesMask = None,
                               flags = 2)

            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            cv2.imshow('fuck_you',img3)
            cv2.waitKey(200)

        return np.float32([kp1[m.queryIdx].pt for m in good]), np.float32([kp2[m.trainIdx].pt for m in good])

    def tsfm_mat(self, r, t):
        z = np.eye(4, dtype=np.float64)
        z[:3, :3] = r
        z[:3, 3] = t
        return z

    
    def decomp_essential_mat(self, e, q1, q2): 
        def sum_z_cal_relative_scale(r, t):
            t = self.tsfm_mat(r, t)
            p = np.matmul(np.concatenate((self.cm, np.zeros((3, 1))), axis=1), t)
            hom_q1 = cv2.triangulatePoints(self.pm, p, q1.T, q2.T)
            hom_q2 = np.matmul(t, hom_q1)

            uhom_q1 = hom_q1[:3, :] / hom_q1[3, :]
            uhom_q2 = hom_q2[:3, :] / hom_q2[3, :]

            sum_of_pos_z_q1 = sum(uhom_q1[2, :] > 0)
            sum_of_pos_z_q2 = sum(uhom_q2[2, :] > 0)

            relative_scale = np.mean(np.linalg.norm(uhom_q1.T[:-1] - uhom_q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_q2.T[:-1] - uhom_q2.T[1:], axis=-1))
            return sum_of_pos_z_q1 + sum_of_pos_z_q2, relative_scale

        r1, r2, t = cv2.decomposeEssentialMat(e)
        t = np.squeeze(t)

        pairs = [[r1, t], [r1, -t], [r2, t], [r2, -t]]

        z_sums = []
        relative_scales = []
        for r, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(r, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        r1, t = right_pair
        t = t * relative_scale

        return [r1, t]

    def get_pose(self, q1, q2):
        e, _ = cv2.findEssentialMat(q1, q2, self.cm, threshold=1)
        [r, t] = self.decomp_essential_mat(e, q1, q2) 
        return self.tsfm_mat(r, t.flatten())



