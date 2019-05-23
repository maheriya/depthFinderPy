import numpy as np
import cv2
import glob
import os
import time
import argparse
import sys

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        # number of inner black chess patterns (8,6) has [7,5] pattern sizes
        self.patternsize = [9, 6]
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) 9, 6 interior number of corners
        self.objp = np.zeros((self.patternsize[0] * self.patternsize[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.patternsize[0], 0:self.patternsize[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        print('self.cal_path: {}'.format(self.cal_path))
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(os.path.join(cal_path, 'RIGHT/*.png'))
        images_left = glob.glob(os.path.join(cal_path, 'LEFT/*.png'))
        images_left.sort()
        images_right.sort()
        img_shape = None
        print('images_right: {}'.format(images_right))
        #time.sleep(3)

        # for i in range(4):
        for i, fname in enumerate(images_right):
            print('i: {}'.format(i))
            img_l = cv2.imread(images_left[i])
            print('images_left[i]: {}'.format(images_left[i]))
            img_r = cv2.imread(images_right[i])
            print('images_right[i]: {}'.format(images_right[i]))

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.patternsize[0], self.patternsize[1]), None)
            print('ret_l: {}'.format(ret_l))
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.patternsize[0], self.patternsize[1]), None)
            print('ret_r: {}'.format(ret_r))
            if not ret_r:
                continue  ## continue, don't break -- there will be other images to look at
                #break
            if not ret_l:
                continue  ## continue, don't break -- there will be other images to look at
                #break

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (self.patternsize[0], self.patternsize[1]), corners_l, ret_l)

            if ret_r:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

            # Draw and display the corners
            ret_r = cv2.drawChessboardCorners(img_r, (self.patternsize[0], self.patternsize[1]), corners_r, ret_r)
            numpy_horizontal = np.hstack((ret_l, ret_r))
            cv2.imshow("chess", numpy_horizontal)
            cv2.moveWindow("chess", 2600, 600)
            cv2.waitKey(300)
            img_shape = gray_l.shape[::-1]

        if img_shape:
            print('img_shape: {}'.format(img_shape))

            rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_l, img_shape, None, None)
            rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_r, img_shape, None, None)
        else:
            print("img_shape is None")
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('K1 M1 Intrinsic_mtx_1', M1)
        print('K2 M2 Intrinsic_mtx_2', M2)
        print('dist_1', d1)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        sfile = "intrinsics.yml"
        savefile = cv2.FileStorage(sfile, cv2.FILE_STORAGE_WRITE)
        if not savefile.isOpened():
            print("Could not open file {} for saving results".format(sfile))
            sys.exit()
        savefile.writeComment("This file contains Left and Right camera intrinsic parameters\n")
        savefile.write("M1", M1)
        savefile.write("D1", d1)
        savefile.write("M2", M2)
        savefile.write("D2", d2)
        savefile.release()
        
        sfile = "extrinsics.yml"
        savefile = cv2.FileStorage(sfile, cv2.FILE_STORAGE_WRITE)
        if not savefile.isOpened():
            print("Could not open file {} for saving results".format(sfile))
            sys.exit()
        savefile.writeComment("This file contains extrinsic matrices for Left and Right setup\n")
        savefile.write("R", R)
        savefile.write("T", T)
        savefile.release()
        

        cv2.destroyAllWindows()
        return camera_model


"""
P = Intrinsic_mtx x Extrinsic_mtx
X3D = P*X_image
P = K x [R|t]
K3x3 = Calibration matrix
R3x3 = Rotation matrix
t3x1 = Translation matrix
F = Fundamental matrix
distortion camera 1
distortion camera 2


3D Translation (I|t)
Extrinsic_mtx = [R|t] = 3D Translation x 3D Rotation
Mp = Mint * Mext

Xc, Yc, Zc new coordinates on the world coordinates
R3x3 = Rotation matrix
Xw, Yw, Zw original coordinates on the world cooridnates
                      [    
                      [R3x3] [T3x1]
[[Xc] [Yc] [Zc] [1]]  =    [0]    [1]    * [[Xw] [Yw] [Zw] [1]]
                          
                          
===========================================
Chessboard calibration results:
===========================================
                                   ]
K1 M1 Intrinsic_mtx_1 
[[558.20202708   0.         648.77029399]
 [  0.         431.58306168 527.96911721]
 [  0.           0.           1.        ]]
K2 M2 Intrinsic_mtx_2 
[[362.6092258    0.         640.04547901]
 [  0.         442.21617711 511.98875488]
 [  0.           0.           1.        ]]
dist_1 [[ 0.02480611 -0.01039774  0.02295238 -0.02049286 -0.02267037]]
dist_2 [[ 3.57372897e-05 -8.56071883e-06  1.62101038e-05 -3.63453938e-05 -6.55436816e-06]]
R [[ 0.98604545 -0.13932507  0.09112022]
 [ 0.13376779  0.98891032  0.06451788]
 [-0.09909868 -0.05142861  0.99374773]]
t [[-55.21990115]
 [ -2.67028115]
 [  3.99748918]]
E [[ -0.27011394  -3.81582946  -2.91149535]
 [ -1.53051355  -3.39683331  55.23890345]
 [ -4.75362547 -54.97956726  -3.31935433]]
F [[ 2.66053330e-07  4.86113497e-06 -1.13836561e-03]
 [ 1.23612705e-06  3.54835597e-06 -2.75789995e-02]
 [ 8.94626663e-04  2.04693126e-02  1.00000000e+00]]


"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
    # cal_data = StereoCalibration("/mnt/R931GB/basler_test3/")
    # print('cal_data: {}'.format(cal_data))

