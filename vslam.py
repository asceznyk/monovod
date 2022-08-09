import cv2
import numpy as np

class vSLAM():
    def __init__(self, cm):
        self.orb = cv2.ORB_create(3000) 
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.frames = []
        self.cm = cm

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

    def get_pose(self, q1, q2):
        e, _ = cv2.findEssentialMat(q1, q2, self.cm, threshold=1)
        r, _, t = cv2.decomposeEssentialMat(e) 
        return self.tsfm_mat(r, t.flatten())



