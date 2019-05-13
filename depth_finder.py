'''
For a given set of (x,y) coordinates for left and right views, compute world coordinates (X,Y,Z)

'''
# Python 2/3 compatibility
from __future__ import print_function

import sys
import argparse
import numpy as np
import cv2 as cv

# Global variables. Fixed for BATFAST cameras
pixsize = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size = (1280, 1024)

DEBUG = 0

class depthFinder:
    def __init__(self, intrinsics, extrinsics, rectify):
        fmt = lambda x: "%10.3f" % x
        np.set_printoptions(formatter={'float_kind':fmt})

        self.rectify = rectify
        ## Read calibration files
        fsi = cv.FileStorage(intrinsics, cv.FILE_STORAGE_READ)
        fse = cv.FileStorage(extrinsics, cv.FILE_STORAGE_READ)
        self.M1 = fsi.getNode('M1').mat()
        self.D1 = fsi.getNode('D1').mat()
        self.M2 = fsi.getNode('M2').mat()
        self.D2 = fsi.getNode('D2').mat()
        fx = self.M1[0,0]
        fy = self.M1[1,1]
        f  = (fx+fy) * pixsize / 2.
        if DEBUG:
            print("Camera f = {:.2f}mm [For sanity check]".format(f))
        ## Extrinsics
        self.R  = fse.getNode('R').mat()
        self.T  = fse.getNode('T').mat()
        
        if DEBUG: print("Performing rectification")
        # Create rectification matrices (R1, R2, P1, P2, Q)
        self.R1, self.R2, self.P1, self.P2,  Q, roi1, roi2 = cv.stereoRectify(
            self.M1, self.D1, self.M2, self.D2, img_size, self.R, self.T,  ## Inputs
            alpha=-1, flags=0)
            

    def get3D(self, l, r):
        if DEBUG:
            print("Left point: {}".format(l))
            print("Right point: {}".format(r))
        lpoint = np.array([[l]], dtype=np.float32)
        rpoint = np.array([[r]], dtype=np.float32)
        unl = cv.undistortPoints(src=lpoint, cameraMatrix=self.M1, distCoeffs=self.D1, R=self.R1, P=self.P1)
        unr = cv.undistortPoints(src=rpoint, cameraMatrix=self.M2, distCoeffs=self.D2, R=self.R2, P=self.P2)
        if DEBUG:
            print("Left undistorted point: {}".format(unl))
            print("Right undistorted point: {}".format(unr))


        points4D = cv.triangulatePoints(projMatr1=self.P1, projMatr2=self.P2, projPoints1=unl, projPoints2=unr)
        point3D = cv.convertPointsFromHomogeneous(points4D.transpose()).squeeze()
        if DEBUG:
            print("point3D: {}".format(point3D))
        
        ## Apply necessary coordinate shift
        if 0:
            ## It is possible to shift or rotate the coordinates here to get desired new world coordinate center
            np3D = np.dot(point3D.transpose(), self.R1) + np.array([0,0,-320.], dtype=np.float32)
        else:
            np3D = point3D

        return np3D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Find world coordinates from a pair of image correspondence points. For example:
        python3 depth_finder --lpoint 439. 527. --rpoint 443. 507.

    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--intrinsics', type=str, default="data/calib/intrinsics.yml",
                        help='YAML camera intrinsics calibration file')
    parser.add_argument('--extrinsics', type=str, default="data/calib/extrinsics.yml",
                        help='YAML camera extrinsics calibration file')
    parser.add_argument('--rectify', action="store_true", default=False,
                        help='If option is specified, perform rectification. Default is not to perform rectification.')
    reqargs = parser.add_argument_group('required arguments')
    reqargs.add_argument('--lpoint', nargs=2, type=float, required=True,
                        help='Coordinates of point on left image.')
    reqargs.add_argument('--rpoint', nargs=2, type=float, required=True,
                        help='Coordinates of point on right image.')

    args = parser.parse_args()

    try:
        res1 = len(args.lpoint)
        res2 = len(args.rpoint)
    except:
        print("Please provide input points")
        parser.print_help()
        sys.exit()
    else:
        if (res1 != 2 and res2 != 2):
            print("Please provide input points")
            parser.print_help()
            sys.exit()

    if (args.rectify):
        print("Rectification will be performed")
    else:
        print("Rectification will NOT be performed")
    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depthFinder(args.intrinsics, args.extrinsics, args.rectify)
     
    ## Compute world coordinates from 2D image points
    point3d = df.get3D(args.lpoint, args.rpoint)
    if DEBUG:
        ostr = '3D world coordinates are: '
    else:
        ostr = ''
    print('{}'.format(point3d))
