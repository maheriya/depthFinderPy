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
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"

    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depth_finder.depthFinder(intrinsics, extrinsics, rectify=True)


    rpoints = []
    ## With new camera position
    ## Left  R0: 415,533    641,519     796,510
    ## Left  R1: 425,626    684,612     860,598
    ## Right R0: 424,502    642,510     801,517
    ## Right R1: 355,588    599,604     783,611
    ##
    ## L Coordinates 1: R0,C0             // L+R Coordinate 2: R0,C1             // L+R Coordinate 3: R0,C2
    lpoints = np.array([[415.,533.], [641.,519.], [796.,510.],
                        [425.,626.], [684.,612.], [860.,598.]], dtype=np.float32)
    rpoints = np.array([[424.,502.], [642.,510.], [801.,517.],
                        [355.,588.], [599.,604.], [783.,611.]], dtype=np.float32)

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
    print("Estimated Z: ", estz)
    print("Measured  Z: ", mz)
    errors = estz - mz;
    print("errors    Z: ", errors)
    rms = np.sqrt(np.dot(errors, errors)/len(errors))
    print("RMS error: {:.0f}".format(rms))
    ## Result of above code:
    # Point 0: [  201    70  8345]
    # Point 1: [ 1302    63  8349]
    # Point 2: [ 2078    62  8336]
    # Point 3: [  195   455  7293]
    # Point 4: [ 1296   463  7268]
    # Point 5: [ 2080   455  7274]
    # Estimated Z:  [ 8345  8349  8336  7293  7268  7274]
    # Measured  Z:  [ 8036  8129  8290  6886  6996  7190]
    # errors    Z:  [  309   220    46   407   272    84]
    # RMS error: 256





#EOF