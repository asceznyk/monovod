import cv2
import numpy as np

from utils import *

class Map():
    def __init__(self):
        self.points = []
        self.poses = []

    def add_points(self, points):
        self.points.append(points)

    def add_pose(self, pose):
        self.poses.append(pose)

class MONOVOD():
    def __init__(self, calibs_path=None, f=716, n_feat=3000):
        self.orb = cv2.ORB_create(n_feat) 
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.frames = []
        self.f = f
        self.pm, self.cm = None, None
        if calibs_path is not None:
            self.pm, self.cm = read_calib(calibs_path)
            print(self.pm)

        self.mapp = Map() 

    def fill_calib(self):
        self.pm, self.cm = init_cam_intrinsics(self.frames[-1], self.f) 
        print(self.pm)

    def add_frame(self, f):
        self.frames.append(f)
        return len(self.frames)-1

    def get_matches(self, draw_matches=0):
        kp1, des1 = self.orb.detectAndCompute(self.frames[-2], None) 
        kp2, des2 = self.orb.detectAndCompute(self.frames[-1], None)
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

            img3 = cv2.drawMatches(self.frames[-2], kp1, self.frames[-1], kp2, good, None, **draw_params)
            cv2.imshow('doesnt_matter',img3)
            cv2.waitKey(200)

        return np.float32([kp1[m.queryIdx].pt for m in good]), np.float32([kp2[m.trainIdx].pt for m in good])
        
    def tsfm_mat(self, r, t):
        z = np.eye(4, dtype=np.float64)
        z[:3, :3] = r
        z[:3, 3] = t
        return z

    def project_points(self, pose):
        pts = self.mapp.points[-1]
        pts = np.vstack((pts, np.ones((1, pts.shape[1]))))
        self.mapp.points[-1] = np.matmul(pose, pts)[:3] 
    
    def calc_rt(self, e, q1, q2): 
        def sum_z_cal_relative_scale(r, t):
            x = self.tsfm_mat(r, t) 
            hom_q1 = triangulate_points(self.pm, np.matmul(self.pm, x), q1, q2) 
            hom_q2 = np.matmul(x, hom_q1)

            uhom_q1 = hom_q1[:3, :] / hom_q1[3, :]
            uhom_q2 = hom_q2[:3, :] / hom_q2[3, :]

            sum_of_pos_z_q1 = sum(uhom_q1[2, :] > 0)
            sum_of_pos_z_q2 = sum(uhom_q2[2, :] > 0)

            relative_scale = np.mean(
                np.linalg.norm(uhom_q1.T[:-1] - uhom_q1.T[1:], axis=-1) / 
                np.linalg.norm(uhom_q2.T[:-1] - uhom_q2.T[1:], axis=-1)
            )

            return uhom_q1, sum_of_pos_z_q1 + sum_of_pos_z_q2, relative_scale

        r1, r2, t = cv2.decomposeEssentialMat(e)
        t = np.squeeze(t)

        pairs = [[r1, t], [r1, -t], [r2, t], [r2, -t]]
        sumzs = []
        for i, [r, t] in enumerate(pairs): 
            sumzs.append((i, *sum_z_cal_relative_scale(r, t)))

        j, pts, _, s = max(sumzs, key=lambda x: x[2])
        r, t = pairs[j]
        self.mapp.add_points(pts)
        return [r, s*t]

    def get_pose(self, q1, q2):
        e, _ = cv2.findEssentialMat(q1, q2, self.cm, threshold=1)
        [r, t] = self.calc_rt(e, q1, q2)
        return self.tsfm_mat(r, np.squeeze(t)) 


