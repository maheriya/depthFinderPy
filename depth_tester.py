'''
Test the depth_finder.py script with measured world 3D points

'''
# Python 2/3 compatibility
from __future__ import print_function

import sys
import argparse
import numpy as np
import cv2 as cv
import depth_finder

# Global variables. Fixed for BATFAST cameras
pixsize = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size = (1280, 1024)

DEBUG = 1

if __name__ == '__main__':
    fmt = lambda x: "%5.0f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    print("-------------------------------------------------------------------")
    print("All results are in mm scale\n\n")
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"

    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depth_finder.depthFinder(intrinsics, extrinsics, rectify=True)


    ##-#######################################################################################
    ## Test 1: 6 balls on the ground.
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 6 balls
    ##  World coordinates: depth Z of each ball, and distances between balls
    ##-#######################################################################################
    rpoints = []
    ## With new camera position
    ## Left  R0: 415,533    641,519     796,510
    ## Left  R1: 425,626    684,612     860,598
    ## Right R0: 424,502    642,510     801,517
    ## Right R1: 355,588    599,604     783,611
    ##
    ## L Coordinates :  R0,C0
    lpoints = np.array([[415.,533.], [641.,519.], [796.,510.],                      ## Left  p00, p01, p11
                        [425.,626.], [684.,612.], [860.,598.]], dtype=np.float32)   ## Left  p10, p11, p12
    ## R Coordinate : R0,C2
    rpoints = np.array([[424.,502.], [642.,510.], [801.,517.],                      ## Right p00, p01, p11
                        [355.,588.], [599.,604.], [783.,611.]], dtype=np.float32)   ## Right p10, p11, p12

    p3D = [df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ]
    for i in range(len(p3D)):
        print("Point {}: {}".format(i, p3D[i]))

    mz = np.array(
            [8036.,                    # z00
             8129.,                    # z01
             8290.,                    # z02
             6886.,                    # z10
             6996.,                    # z11
             7190.], dtype=np.float32) # z12
    estz = np.array([p[2] for p in p3D])
    print("\n\n---------------------------------------------------")
    print("Test 1, Results A: Z measurements for all 6 balls\n")
    print("Estimated Z: ", estz)
    print("Measured  Z: ", mz)
    errors = estz - mz;
    print("Errors in Z: ", errors)
    rms = np.sqrt(np.dot(errors, errors)/len(errors))
    print("RMS error: {:.0f}".format(rms)) ## COMMENT: Currently this is large: 256
    ## Result of above code:
    # Point 0: [  201    70  8345]
    # Point 1: [ 1302    63  8349]
    # Point 2: [ 2078    62  8336]
    # Point 3: [  195   455  7293]
    # Point 4: [ 1296   463  7268]
    # Point 5: [ 2080   455  7274]
    # Estimated Z:  [ 8345  8349  8336  7293  7268  7274]
    # Measured  Z:  [ 8036  8129  8290  6886  6996  7190]
    # errors in Z:  [  309   220    46   407   272    84]
    # RMS error: 256

    ##-#######################################################################################
    ## Test 1, Results B : Ball distances
    ##-#######################################################################################
    p00  = p3D[0]
    pref = p3D[1] # Reference point
    p02  = p3D[2]
    p10  = p3D[3]
    p11  = p3D[4]
    p12  = p3D[5]

    d00 = cv.norm(pref-p00);
    d02 = cv.norm(pref-p02);
    d10 = cv.norm(pref-p10);
    d11 = cv.norm(pref-p11);
    d12 = cv.norm(pref-p12);
    print("\n\n---------------------------------------------------")
    print("Test 1, Results B: Relative distance measurements between balls\n")
    print("Estimated Distances: cv.norm(refball-otherball); real (X,Y,Z)):")
    print("d00  : {:.0f}".format(d00))
    print("d02  : {:.0f}".format(d02))
    print("d10  : {:.0f}".format(d10))
    print("d11  : {:.0f}".format(d11))
    print("d12  : {:.0f}".format(d12))
    # Expected real world distances from p01 (pref) represented by . below:
    # m00: 1088      .     m01:  775
    # m10: 1579  m11: 1147 m12: 1385
    m00 = 1088.;
    m02 = 775.;
    m10 = 1579.;
    m11 = 1147.;
    m12 = 1385.;
    print("Measured Distances:")
    print("m00  : {:.0f}".format(m00))
    print("m02  : {:.0f}".format(m02))
    print("m10  : {:.0f}".format(m10))
    print("m11  : {:.0f}".format(m11))
    print("m12  : {:.0f}".format(m12))

    e00 = (m00 - d00);
    e02 = (m02 - d02);
    e10 = (m10 - d10);
    e11 = (m11 - d11);
    e12 = (m12 - d12);
    errors = np.array([e00, e02, e10, e11, e12], dtype=np.float32)
    print("Errors:")
    print("e00  : {:.0f}".format(e00))
    print("e02  : {:.0f}".format(e02))
    print("e10  : {:.0f}".format(e10))
    print("e11  : {:.0f}".format(e11))
    print("e12  : {:.0f}".format(e12))
    erms = np.sqrt(np.dot(errors, errors)/len(errors))
    print("RMS error: {:.0f}".format(erms))  ## COMMENT: This is quite small. This means that our X,Y,Z are correct, 
    #                                                    just that the absolute (0,0,0) are not at a known accurate position
    #                                                    at the center of the camera sensor 


#EOF