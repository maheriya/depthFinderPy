'''
Test the depth_finder.py script with measured world 3D points
This is the final test with 16 balls measurement for verifying the depthFinder
Additional two balls measurement data is the test data (X,Y,Z) unknown to this script,
but known to client.
'''
import sys
import argparse
import numpy as np
import cv2 as cv
import math
from depth_finder import depthFinder

DEBUG = 1


if __name__ == '__main__':
    fmt = lambda x: "%5.0f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    print("-------------------------------------------------------------------\n")
    print("All results are in mm scale")
    print("Origin [0,0,0] is at right cage wall bottom")
    print("X axis parallel to left and right cameras baseline, increases towards left")
    print("Y axis perpendicular to ground, with ground as 0")
    print("Z axis parallel to ground, with camera baseline as 0, increasing away from cameras")
    print("-------------------------------------------------------------------\n")
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"
    posefile = 'data/calib/pose_Z5300_2.yml' # Chessboard in the center of cage
    df = depthFinder(intrinsics, extrinsics, shift=True, offset=np.float32([0,0,0]), posefile=posefile) ## For now, don't shift origin -- we are testing it
    #fsi = cv.FileStorage(posefile, cv.FILE_STORAGE_READ)
    #if not fsi.isOpened():
    #    print("Could not open file {} for reading calibration data".format(posefile))
    #    sys.exit()
    #
    # Left              Right                   X,Y,Z Real World
    # Left column (5 balls)
    # [ 299.0,  477.0], [ 373.0,  443.0],       [3385,    0, 8905],
    # [ 291.0,  544.0], [ 310.0,  505.0],       [3385,    0, 7905],
    # [ 284.0,  628.0], [ 238.0,  578.0],       [3385,    0, 6905],
    # [ 282.0,  733.0], [ 153.0,  669.0],       [3385,    0, 5905],
    # [ 268.0,  870.0], [  43.0,  780.0],       [3385,    0, 4905],
    #                                           
    # Middle column (6 balls)                   
    # [ 631.0,  463.0], [ 689.0,  456.0],       [1600,    0, 8905],
    # [ 661.0,  527.0], [ 661.0,  522.0],       [1600,    0, 7905],
    # [ 702.0,  608.0], [ 629.0,  603.0],       [1600,    0, 6905],
    # [ 752.0,  708.0], [ 584.0,  705.0],       [1600,    0, 5905],
    # [ 817.0,  835.0], [ 528.0,  834.0],       [1600,    0, 4905],
    # [ 902.0, 1006.0], [ 450.0, 1009.0],       [1600,    0, 3905],
    # 
    # Right column (5 balls)
    # [ 907.0,  451.0], [ 979.0,  466.0],       [  35,    0, 8905],
    # [ 970.0,  510.0], [ 987.0,  534.0],       [  35,    0, 7905],
    # [1041.0,  585.0], [ 993.0,  618.0],       [  35,    0, 6905],
    # [1127.0,  673.0], [ 999.0,  720.0],       [  35,    0, 5905],
    # [1236.0,  789.0], [1008.0,  861.0],       [  35,    0, 4905],
    # 
    # Ball in the air (1 ball)
    # [1005.0,  222.0], [ 844.0,  223.0],       [ 676, 1874, 5810]

    # Construct lpoints and rpoints from above data
    lpoints = np.float32([
        # Left column (5 balls)
        [ 299.0,  477.0],
        [ 291.0,  544.0],
        [ 284.0,  628.0],
        [ 282.0,  733.0],
        [ 268.0,  870.0],

        # Middle column (6 balls)                   
        [ 631.0,  463.0],
        [ 661.0,  527.0],
        [ 702.0,  608.0],
        [ 752.0,  708.0],
        [ 817.0,  835.0],
        [ 902.0, 1006.0],

        # Right column (5 balls)
        [ 907.0,  451.0],
        [ 970.0,  510.0],
        [1041.0,  585.0],
        [1127.0,  673.0],
        [1236.0,  789.0],

        ])

    rpoints = np.float32([
        # Left column (5 balls)
        [ 373.0,  443.0],
        [ 310.0,  505.0],
        [ 238.0,  578.0],
        [ 153.0,  669.0],
        [  43.0,  780.0],

        # Middle column (6 balls)
        [ 689.0,  456.0],
        [ 661.0,  522.0],
        [ 629.0,  603.0],
        [ 584.0,  705.0],
        [ 528.0,  834.0],
        [ 450.0, 1009.0],

        # Right column (5 balls)
        [ 979.0,  466.0],
        [ 987.0,  534.0],
        [ 993.0,  618.0],
        [ 999.0,  720.0],
        [1008.0,  861.0],

        ])

    # X,Y,Z Real World  
    opoints = np.float32([
        # Left column (5 balls)
        [3385,   35, 8905],
        [3385,   35, 7905],
        [3385,   35, 6905],
        [3385,   35, 5905],
        [3385,   35, 4905],
                          
        # Middle column (6 balls)
        [1600,   35, 8905],
        [1600,   35, 7905],
        [1600,   35, 6905],
        [1600,   35, 5905],
        [1600,   35, 4905],
        [1600,   35, 3905],
                          
        # Right column (5 balls)
        [  35,   35, 8905],
        [  35,   35, 7905],
        [  35,   35, 6905],
        [  35,   35, 5905],
        [  35,   35, 4905],

        ])

    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0])])
    print("depthFinder output\tExpected result\tDifference")
   
    errs = np.float32([])
    for i in range(len(p3D)):
        err = np.float32(opoints[i]-p3D[i])
        errs = np.append(errs, err, axis=0)
        print("{}\t{}\t{}".format(p3D[i], opoints[i], err))
    rms = np.sqrt(np.dot(errs, errs)/len(errs))
    print("RMS: ", rms)
    ## Output
    ## depthFinder output     Expected result        Difference
    ## [ 3388    42  8931]    [ 3385    35  8905]    [   -3    -7   -26]
    ## [ 3398    43  7896]    [ 3385    35  7905]    [  -13    -8     9]
    ## [ 3399    40  6896]    [ 3385    35  6905]    [  -14    -5     9]
    ## [ 3375    50  5878]    [ 3385    35  5905]    [   10   -15    27]
    ## [ 3395    47  4903]    [ 3385    35  4905]    [  -10   -12     2]
    ## [ 1597    45  8903]    [ 1600    35  8905]    [    3   -10     2]
    ## [ 1600    30  7922]    [ 1600    35  7905]    [   -0     5   -17]
    ## [ 1589    19  6925]    [ 1600    35  6905]    [   11    16   -20]
    ## [ 1590    25  5906]    [ 1600    35  5905]    [   10    10    -1]
    ## [ 1586    35  4908]    [ 1600    35  4905]    [   14    -0    -3]
    ## [ 1588    44  3896]    [ 1600    35  3905]    [   12    -9     9]
    ## [   37    54  8901]    [   35    35  8905]    [   -2   -19     4]
    ## [   21    60  7873]    [   35    35  7905]    [   14   -25    32]
    ## [   19    49  6879]    [   35    35  6905]    [   16   -14    26]
    ## [   22    60  5889]    [   35    35  5905]    [   13   -25    16]
    ## [   16    54  4887]    [   35    35  4905]    [   19   -19    18]
    ## RMS:  14.438768911815234

    
    # Ball in the air (1 ball)
    # Left              Right                   X,Y,Z Real World
    # [1005.0,  222.0], [ 844.0,  223.0],       [ 676, 1874, 5810]
    lpoints = np.float32([
        [1005.0,  222.0],
        ])
    rpoints = np.float32([
        [ 844.0,  223.0],
        ])
    # X,Y,Z Real World  
    opoints = np.float32([
        [ 676, 1874, 5810],
        ])
    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0])])
    print("depthFinder output\tExpected result\tDifference")
   
    errs = np.float32([])
    for i in range(len(p3D)):
        err = np.float32(opoints[i]-p3D[i])
        errs = np.append(errs, err, axis=0)
        print("{}\t{}\t{}".format(p3D[i], opoints[i], err))
    rms = np.sqrt(np.dot(errs, errs)/len(errs))
    print("RMS: ", rms)
    ## Output:
    ## depthFinder output     Expected result        Difference
    ## [  664  1888  5977]    [  676  1874  5810]    [   12   -14  -167]



