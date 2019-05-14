'''
Test the depth_finder.py script with measured world 3D points
This is test 2. 
Uses calibration image to align output X,Y,Z as an example of changing the coordinates of
the output X,Y,Z. This transform is done on top of real world coordinates output by the 
depth algorithm (without changing the depth algorithm) by simply selecting an arbitrary 
3D point (in this case, the top-left corner of the chessboard pattern), and making that 
the origin (0,0,0) instead of the camera. After that all the X,Y,Z values will reflect 
this corner as origin (as a result, Z won't change; also X and Y will change as you would 
move left/right/top/down on the chessboard.

The accuracy of the algorithm seems to be a few millimeters (less than 1cm).
 
'''
# Python 2/3 compatibility
from __future__ import print_function

import sys
import argparse
import numpy as np
import cv2 as cv
import math
import depth_finder

# Global variables. Fixed for BATFAST cameras
pixsize = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size = (1280, 1024)
squareSize = 114.8
DEBUG = 1

if __name__ == '__main__':
    fmt = lambda x: "%5.3f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    print("-------------------------------------------------------------------")
    print("All results are in mm scale\n\n") ## This is embedded in the calibration matrices
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"

    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depth_finder.depthFinder(intrinsics, extrinsics)


    ##-#######################################################################################
    ## Test 2: Calibration target. Find coordinates of five corners
    ## Available measurements: 
    ##  Image coordinates: (x,y) of 5 corners on calibration image {left|right}-035
    ##  World coordinates: distances between corners
    fmt = lambda x: "%5.0f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    rpoints = []
    ## Chessboard corners:
    ## Left:    0          4           8
    ##   0   545.,443.              755.,428.            # 
    ##   3              652.,512.                        # Center point; not used right now
    ##   5   549.,572.              754.,554.            # 
    ## Right                                             # 
    ##   0   481.,426.              693.,429.            # 
    ##   3              587.,504.                        # Center point; not used right now
    ##   5   486.,552.              693.,557.            # 
    ##
    lpoints = np.array([[545.,443.], [755.,428.],                      ## Left  p00, p01, p01
                        [549.,572.], [754.,554.]], dtype=np.float32)   ## Left  p10, p11, p12
    rpoints = np.array([[481.,426.], [693.,429.],                      ## Right p00, p01, p02
                        [486.,552.], [693.,557.]], dtype=np.float32)   ## Right p10, p11, p12

    ## Rotation vector that record camera pose relative to the chessboard
    rvec  = np.float32([ 0.305, -0.05, -0.012])  ## Adjusted rvec

    Rt, _ = cv.Rodrigues(rvec)
    R = Rt.T
    ## Find world coordinates of the l+r (x,y) points
    p3D = [df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ]
    ## Transform new 3D points to change their coordinate system to be the top-left corner of chessboard
    np3D = [np.dot((p3D[i]-p3D[0]), Rt) for i in range(len(p3D)) ]
    print(" Pt {}: {}".format(0, p3D[0]))
    print("00 {}\t\t01 {}".format(np3D[0], np3D[1]))
    print("10 {}\t\t11 {}".format(np3D[2], np3D[3]))

    ## This should result in following output (four corners of chessboard):
    ## Note how X,Y,Z behave relative to the board's axes. We have simply transformed (rotate+translate) the 3D coordinates
    ## Depth algorithm doesn't change. This can be done for any 3D point of choice and rotation
    ## 00 [    0     0     0]        01 [  918    -5    -4]
    ## 10 [    2   579     2]        11 [  919   573    -4]



    ##-#######################################################################################
    ## Test 1: Corners distances
    ##-#######################################################################################
    p00  = np3D[0]  ## Reference
    p01  = np3D[1]
    p10  = np3D[2]
    p11  = np3D[3]
    pref = p00

    d01 = cv.norm(pref-p01);
    d10 = cv.norm(pref-p10);
    d11 = cv.norm(pref-p11);
    print("\n\n---------------------------------------------------")
    print("Test 1: Relative distance measurements between chessboard corners\n")
    print("Estimated Distances: cv.norm(refball-otherball); real (X,Y,Z)):")
    print("d01  : {:.0f}".format(d01))
    print("d10  : {:.0f}".format(d10))
    print("d11  : {:.0f}".format(d11))
    # Expected real world distances from p00 (pref) corner (each square is 115mm wide):
    m01 = squareSize * 8.;
    m10 = squareSize * 5.;
    m11 = math.sqrt(m01*m01 + m10*m10) ## m11 happens to be the hypotenuse 
    print("Measured Distances:")
    print("m01  : {:.0f}".format(m01))
    print("m10  : {:.0f}".format(m10))
    print("m11  : {:.0f}".format(m11))

    e01 = (m01 - d01);
    e10 = (m10 - d10);
    e11 = (m11 - d11);
    errors = np.array([e01, e10, e11], dtype=np.float32)
    print("Errors:")
    print("e01  : {:.0f}".format(e01))
    print("e10  : {:.0f}".format(e10))
    print("e11  : {:.0f}".format(e11))
    erms = np.sqrt(np.dot(errors, errors)/len(errors))
    print("RMS error: {:.0f}".format(erms))  ## COMMENT: This is quite small. This means that our X,Y,Z are correct, 
    #                                                    just that the absolute (0,0,0) are not at a known accurate position
    #                                                    at the center of the camera sensor 

#EOF