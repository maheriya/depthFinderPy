'''
For a given set of (x,y) coordinates for left and right views, compute world coordinates (X,Y,Z)

'''
import argparse
import numpy as np
import cv2 as cv
import sys

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    sys.exit(1)

#-############################################################################################
# Global parameters. Fixed for BATFAST cameras
#-############################################################################################
pixsize    = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size   = (1280, 1024)    ## Camera sensor size in horizontal and vertical pixels
pwidth     = 9               ## Number of corners in horizontal direction (p stands for pattern)
pheight    = 6               ## Number of corners in vertical direction
squareSize = 114.9           ## Size of squares of chessboard pattern
ydir       = -1.             ## Make it -1 to invert Y axis to make it Y-up. OpenCV is Y-down, right handed
#-############################################################################################
DEBUG = 0

class depthFinder:
    def __init__(self, intrinsics, extrinsics, shift, offset, posefile):
        self.shift = shift
        self.offset = offset

        ## Read calibration files
        fsi = cv.FileStorage(intrinsics, cv.FILE_STORAGE_READ)
        fse = cv.FileStorage(extrinsics, cv.FILE_STORAGE_READ)
        self.M1 = fsi.getNode('M1').mat()
        self.D1 = fsi.getNode('D1').mat()
        self.M2 = fsi.getNode('M2').mat()
        self.D2 = fsi.getNode('D2').mat()
        fsi.release()
        fx = self.M1[0,0]
        fy = self.M1[1,1]
        f  = (fx+fy) * pixsize / 2.
        if DEBUG:
            print("Camera f = {:.2f}mm [For sanity check]".format(f))
        ## Extrinsics
        self.R  = fse.getNode('R').mat()
        self.T  = fse.getNode('T').mat()
        fse.release()

        # Get pose for origin shift
        fsp = cv.FileStorage(posefile, cv.FILE_STORAGE_READ)
        self.Rshift = fsp.getNode('R').mat()
        self.Tshift = fsp.getNode('T').mat()
        fsp.release()
        
        if DEBUG: print("Performing rectification")
        # Create rectification matrices (R1, R2, P1, P2, Q)
        self.R1, self.R2, self.P1, self.P2,  Q, roi1, roi2 = cv.stereoRectify(
            self.M1, self.D1, self.M2, self.D2, img_size, self.R, self.T,  ## Inputs
            alpha=-1, flags=0)

        

    def shiftOrigin(self, point):
        '''
        Shift origin using rotation and translation vector generated based on calibration target
        earlier.
        Works on a single 3D point.
        '''
        npt = np.dot(self.Rshift, point.transpose()) + self.Tshift
        npt = (offset + npt.transpose()).squeeze(axis=0) # Account for offset
        if (ydir == -1.):
            npt[1] *= ydir ## Make this Y-up coordinate system
        return npt

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
        point3D = cv.convertPointsFromHomogeneous(points4D.transpose()).squeeze(axis=0)
        if DEBUG:
            print("point3D: {}".format(point3D))
        
        ## Apply necessary coordinate shift
        if self.shift:
            return self.shiftOrigin(point3D) 
        else:
            return point3D.squeeze()


class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Find world coordinates from a pair of image correspondence points. For example:
        python3 depth_finder --lpoint 439. 527. --rpoint 443. 507.

    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--intrinsics', type=str, default="data/calib/intrinsics.yml",
                        help='YAML camera intrinsics calibration file')
    parser.add_argument('--extrinsics', type=str, default="data/calib/extrinsics.yml",
                        help='YAML camera extrinsics calibration file')
    parser.add_argument('--pose', dest='posefile', type=str, default="./data/calib/pose.yml",
                        help='File name for saving results')
    parser.add_argument('--enable-shift', action="store_true", default=True,
                        help='Do not shift camera origin. This is provided for calibration purposes')
    parser.add_argument('--shift', '--no-shift', dest='shift', action=NegateAction, nargs=0, default=True,
                        help='Shift [or do not] shift camera origin. This is provided for calibration purposes')
    parser.add_argument('--offset', nargs=3, type=float,
                        help='Real world offset coordinates of the top,left corner point on calibration target. \
                        If provided, the output X,Y,Z are adjusted accordingly. No rotation is performed.')

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
    offset = np.float32([0,0,0])
    try:
        n = len(args.offset)
    except:
        pass
    else:
        if (n == 3):
            offset = np.float32([args.offset[0], args.offset[1], args.offset[2]])
        else:
            print("Please provide offset as a triplet for X Y Z (e.g., --offset 1100 40 4000)")
            parser.print_help()
            sys.exit()

    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depthFinder(args.intrinsics, args.extrinsics, args.shift, offset, args.posefile)

    ## Compute world coordinates from 2D image points
    point3d = df.get3D(args.lpoint, args.rpoint)
    if DEBUG:
        ostr = '3D world coordinates are: '
    else:
        ostr = ''
    print('[{:4.0f} {:4.0f} {:4.0f}]'.format(point3d[0],point3d[1],point3d[2]))
