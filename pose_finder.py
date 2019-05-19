'''
For a given pair of left and right images of a calibration target, compute accurate 
camera pose using 3D points.
Requires depth_finder

'''
import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    sys.exit(1)

import argparse
import numpy as np
import cv2 as cv
import glob
from depth_finder import depthFinder

#-############################################################################################
# Global parameters. Fixed for BATFAST cameras
#-############################################################################################
pixsize    = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size   = (1280, 1024)    ## Camera sensor size in horizontal and vertical pixels
pwidth     = 9               ## Number of corners in horizontal direction (p stands for pattern)
pheight    = 6               ## Number of corners in vertical direction
squareSize = 115             ## Size of squares of chessboard pattern
ydir       = -1.             ## Make it -1 to invert Y axis to make it Y-up. OpenCV is Y-down, right handed
#-############################################################################################
DEBUG = 1

USE_SYNOBJ   = True
FULL_PATTERN = True
AUTO_DET     = True
draw_axis    = False
## For formatted printing of float np arrays
fmt = lambda x: "%12.6f" % x
np.set_printoptions(formatter={'float_kind':fmt})

class poseFinder:
    '''
    This class uses a calibration target image from left and right cameras to calculate rotation
    and translation matrices. These matrices can be then used to shift the origin of the coordinates
    to shift to the top,left corner of the chessboard pattern.
    This class uses depthFinder class to find 3D points of the chessboard pattern.
    The results are saved in pose.yml file.
    '''
    def __init__(self, w, h, intrinsics, left, right, df, sfile, offset):
        '''
        Params:
        w: width in terms of number of corners in chessboard pattern
        h: height in terms of number of corners in chessboard pattern
        intrinsics: YML file that has stored intrinsics
        df: DepthFinder instance. Used to find 3D chess corner points   
        '''
        self.w = w
        self.h = h
        self.left = left
        self.right = right
        self.df = df
        # Load previously saved data
        fsi = cv.FileStorage(intrinsics, cv.FILE_STORAGE_READ)
        if not fsi.isOpened():
            print("Could not open file {} for reading calibration data".format(intrinsics))
            sys.exit()
        fsi.release()

        ## If the target is perpendicular to the ground
        objp = np.zeros((self.w*self.h,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)
        objp = objp * squareSize
        if 1: ## If the target is parallel (or flat) to the ground 
            objp[:,2] = -objp[:,1] ## Z = -Y [Invert for Z axis; Z reduces closer to camera]
            objp[:,1] = 0          ## Y = 0  [Flat on ground
        ## Add offset
        objp += offset

        lpoints = self.findCorners(self.left)
        if (lpoints.shape[0] != (self.w * self.h)): ## Didn't find all corners; no good
            print("Couldn't find all {}*{} ({}) corners in the left view image".format(self.w, self.h, self.w*self.h))
            print("Please provide another calibration target image pair")
            sys.exit(1)
        if DEBUG:
            self.print2D(lpoints)
            limg = cv.imread(self.left)
            cv.drawChessboardCorners(limg, (self.w,self.h), lpoints, True)
            cv.imshow('Left', limg)

        rpoints = self.findCorners(self.right)
        if (rpoints.shape[0] != (self.w * self.h)): ## Didn't find all corners; no good
            print("Couldn't find all {}*{} ({}) corners in the right view image".format(self.w, self.h, self.w*self.h))
            print("Please provide another calibration target image pair")
            sys.exit(1)
        if DEBUG:
            print("rpoints shape", rpoints.shape)
            self.print2D(rpoints)
            rimg = cv.imread(self.right)
            cv.drawChessboardCorners(rimg, (self.w,self.h), rpoints, True)
            cv.imshow('Right', rimg)

        ## Open savefile only if corners are found above to avoid creating an empty file
        savefile = cv.FileStorage(sfile, cv.FILE_STORAGE_WRITE)
        if not savefile.isOpened():
            print("Could not open file {} for saving results".format(sfile))
            sys.exit()
        savefile.writeComment("This file contains rotation and translation matrices to shift origin of 3D points\n")
        savefile.write("objp", objp)
        savefile.write("lpoints", lpoints)
        savefile.write("rpoints", rpoints)

        pts3d = np.mat([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0])])
        if DEBUG:
            print("Corners 3D point cloud:")
            self.print3D(pts3d)
        savefile.writeComment("pts3d is the 3D points cloud generated from lpoints and rpoints without origin shift")
        savefile.write("pts3d", pts3d)

        R, T = self.rigidTransform3D(frpts=pts3d, topts=objp)
        if DEBUG:
            print("R: \n", R)
            print("T: \n", T)
        savefile.write("R", R)
        savefile.write("T", T)
        savefile.release()
        print("Wrote R and T into {}".format(sfile))
        cv.waitKey()
        cv.destroyAllWindows()


    def rigidTransform3D(self, frpts, topts):
        '''
        Find Rigid transform from fpts to tpts 3D point clouds.
        Returns rotation matrix R and translation matrix tvec.
        '''
        assert len(frpts) == len(topts), "Inputs must have the same size"
        assert (frpts.shape[1] == 3) and (topts.shape[1] == 3), \
            "This function operates on 3D points. Input shape is not correct."
    
        N = frpts.shape[0]; # total points
    
        centroid_frpts = np.mat(np.mean(frpts, axis=0))
        centroid_topts = np.mat(np.mean(topts, axis=0))
        if DEBUG:
            print("centroid_frpts shape", centroid_frpts.shape)
            print("centroid_topts shape", centroid_topts.shape)
        # center the point clouds around each centroid
        frpts_ctr = frpts - np.tile(centroid_frpts, (N, 1))
        topts_ctr = topts - np.tile(centroid_topts, (N, 1))
        # Find the main 3x3 covariance matrix
        H = np.dot(frpts_ctr.transpose(), topts_ctr)
        assert (H.shape == (3,3)), "Generated covariance matrix H is not 3x3. Check your input shapes"
 
        ## Use SVD to find decomposed U,S,V matrices
        U, S, Vt = np.linalg.svd(H)
        ## Rotation matrix
        R = np.dot(Vt.transpose(), U.transpose())
    
        # Handle the reflection case
        if (np.linalg.det(R) < 0):
            Vt[2,:] *= -1.
            R = np.dot(Vt.transpose(), U.transpose())
            #R[2,:] *= -1. ## Alternate
            if DEBUG:
                print("Reflection detected (handled)")
            
    
        ## Translation matrix
        tvec = (np.dot(-R, centroid_frpts.transpose())) + centroid_topts.transpose()
        assert (tvec.shape == (3, 1)), "tvec output shape is not correct. Check your input shapes"

        return R, tvec
    
    def findCorners(self, fname):
        '''
        Find w*h chessboard pattern corners in the image
        '''
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (self.w,self.h), None)
        if ret == False:
            print("No corners found in the image {}".format(fname))
            sys.exit(1)
        if DEBUG:
            print("Found corners in image {}".format(fname))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        return corners2.squeeze(axis=1)

    def print2D(self, pts, opts=None, offset=None, ydir=1.):
        '''
        Print 2D chessboard pattern corner points as a grid of sampled points.
        '''
        for r in [0,2,3,5]:
            pstr = ""
            for c in [0,4,8]:
                pstr += "{}, ".format(pts[r*9 + c])
            print(pstr)

    def print3D(self, pts, opts=None, offset=None, ydir=1.):
        '''
        Print 3D chessboard pattern corner points as a grid of sampled points. Can also print
        the difference between two sets of points while considering offset and Y-axis direction
        '''
        if opts is not None: ## Print the difference between pts  and opts
            ## Account for offset and Y-axis direction
            opts = np.float32([1.,ydir,1.]) * (opts+offset)
            npts = pts - opts
        else:
            npts = pts
        for r in [0,2,3,5]:
            pstr = ""
            for c in [0,4,8]:
                pstr += "{}, ".format(npts[r*9 + c])
            print(pstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Find rotation and translation vectors necessary for shifting the origin from camera coordinates system. 
Requires a pair of calibration providing left and right camera views. Results are saved in a file that can be passed to depth_finder.py

    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--intrinsics', type=str, default="data/calib/intrinsics.yml",
                        help='YAML camera intrinsics calibration file')
    parser.add_argument('--extrinsics', type=str, default="data/calib/extrinsics.yml",
                        help='YAML camera extrinsics calibration file')
    parser.add_argument('--w', type=int, default=pwidth,
                        help='Number of corners in horizontal direction on the checkerboard pattern')
    parser.add_argument('--h', type=int, default=pheight,
                        help='Number of corners in vertical direction on the checkerboard pattern')
    parser.add_argument('--left-image', type=str, default="./data/images/left-chessboard-perpendicular.png",
                        help='Path to calibration images')
    parser.add_argument('--right-image', type=str, default="./data/images/right-chessboard-perpendicular.png",
                        help='Path to calibration images')
    parser.add_argument('--savefile-name', dest='sfile', type=str, default="./data/calib/pose.yml",
                        help='File name for saving results')
    parser.add_argument('--offset', nargs=3, type=float,
                        help='Real world offset coordinates of the top,left corner point on calibration target. \
                        Without the offset, the top,left corner is considered the origin of the coordinate system')

    args = parser.parse_args()
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
    print("Using offset (X,Y,Z) of {}".format(offset))

    ## Instantiate depthFinder
    df = depthFinder(args.intrinsics, args.extrinsics, shift=False) ## no origin shift

    fpose = poseFinder(args.w, args.h, args.intrinsics, args.left_image, args.right_image, df, args.sfile, offset)


