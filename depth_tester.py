'''
Test the depth_finder.py script with measured world 3D points

Here we shift origin in the test; depth_finder.py doesn't do origin shift
This is to verify the algorithm itself. It also allows trying out different
offsets
'''

import sys
import argparse
import numpy as np
import cv2 as cv
from depth_finder import depthFinder

#-############################################################################################
# Global parameters. Fixed for BATFAST cameras
#-############################################################################################
pixsize    = 4.8e-06 * 1000  ## Pixel size for scale; mult by 1000 for millimeter scale
img_size   = (1280, 1024)    ## Camera sensor size in horizontal and vertical pixels
pwidth     = 9               ## Number of corners in horizontal direction
pheight    = 6               ## Number of corners in vertical direction
squareSize = 115             ## Size of squares of chessboard pattern
ydir       = -1.             ## Make it -1 to invert Y axis to make it Y-up. OpenCV is Y-down, right handed
#-############################################################################################

## A 9x6 chessboard object points with 115 mm squares.
# def getObjPoints(w, h, squareSize):
#     objp = np.zeros((w*h,3), np.float32)
#     objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
#     objp = objp * squareSize
#     return objp

## A 9x6 chessboard object points with 115 mm squares
def getObjPoints(w, h, squareSize):
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp * squareSize
    return objp

def print3D(pts, opts=None, offset=None, ydir=1.):
    '''
    Print 3D points as grid of sampled points. Can also print difference between
    two sets of points while considering offset and Y-axis direction
    '''
    if opts is not None: ## Print the difference between pts  and opts
        ## Account for offset and Y-axis direction
        opts = np.float32([1.,ydir,1.]) * (opts+offset)
        npts = pts - opts
    else:
        npts = pts
    for r in [0,2,3,5]:
        pstr = ""
        for c in [0,2,4,8]:
            pstr += "{}, ".format(npts[r*9 + c])
        print(pstr)

def getRMSErrors(pts, opts, offset, ydir):
    '''
    Compute RMS errors of 3D distance compared to reference object points
    '''
    err = [cv.norm(pts[i], (np.float32([1.,ydir,1.]) * (opts[i]+offset))) for i in range(len(pts))]
    rms = np.sqrt(np.dot(err, err)/len(pts))
    return rms

def shiftOrigin(Rt, T, point, offset=None):
    '''
    Shift origin using rotation and translation vector generated based on calibration target
    earlier.
    Works on a single 3D point.
    '''
    npt = np.dot(Rt, point.transpose()) + T.transpose()
    if offset is not None:
        npt = offset + npt # Account for offset
    if (ydir == -1.):
        npt[1] *= ydir ## Make this Y-up coordinate system
    return npt

if __name__ == '__main__':
    fmt = lambda x: "%12.6f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    print("\n-------------------------------------------------------------------")
    print("All results are in mm scale")
    print("All manual measurements are from top left cage corner")
    print("-------------------------------------------------------------------\n")
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"

    #posefile = 'data/calib/pose_Z5300_1.yml'  # Chessboard closer to left cage
    posefile = 'data/calib/pose_Z5300_2.yml' # Chessboard in the center of cage
    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depthFinder(intrinsics, extrinsics, shift=False, offset=np.float32([0,0,0]), posefile=posefile) ## For now, don't shift origin -- we are testing it
    fsi = cv.FileStorage(posefile, cv.FILE_STORAGE_READ)
    if not fsi.isOpened():
        print("Could not open file {} for reading calibration data".format(posefile))
        sys.exit()
    R = fsi.getNode('R').mat()
    T = (fsi.getNode('T').mat()).squeeze()
    print("R: \n{}\nT: \n{}".format(R, T))


    ##-#######################################################################################
    ## Test 1: Chessboard perpendicular to ground
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 9 corners manually measured
    ##  World coordinates: (X,Y,Z) of top left corner. Chessboard squares size = 115 mm
    ##-#######################################################################################
    print("\n-------------------------------------------------------------------")
    print(" Test 1 : Chessboard perpendicular to ground")

    ## Create chessboard pattern to compare with
    posefile = 'data/calib/pose_90degree_Z8009.yml' ## Contains l/rpoints for perpendicular chessboard
    fsi = cv.FileStorage(posefile, cv.FILE_STORAGE_READ)
    if not fsi.isOpened():
        print("Could not open file {} for reading calibration data".format(posefile))
        sys.exit()
    lpoints = fsi.getNode('lpoints').mat()
    rpoints = fsi.getNode('rpoints').mat()
    fsi.release()
    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0])])
    print("Corners without origin shift:")
    print3D(p3D)
    npts  = [shiftOrigin(R, T, p, None) for p in p3D]
    print("Corners after origin shift:")
    print3D(npts)
    
    ## Compare with object points
    print("Corners difference after origin shift:")
    
    ## Generate offset for comparison. X and Z are arbitrary since there is no known real location for this board
    offset = np.float32([248, -40-(squareSize*6.+46.), npts[0][2]])
    #offset = np.float32([2, -(squareSize*6.+46.), npts[0][2]])
    print("Offset: ", offset)
    opoints = getObjPoints(pwidth, pheight, squareSize)
    print3D(npts,opoints,offset,ydir)
    rms = getRMSErrors(npts,opoints,offset,ydir)
    print("Total RMS error for all {} points is {}".format(len(npts), rms))

    ##-#######################################################################################
    ## Test 2: Chessboard flat on the ground
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 9 corners manually measured
    ##  World coordinates: (X,Y,Z) of top left corner. Chessboard squares size = 115 mm
    print("\n\n-------------------------------------------------------------------")
    print(" Test 2 : Chessboard flat on the ground")
    lpoints = np.float32([#      0          4          8
                          [396,531], [489,526], [583,520], # 0
                          [396,548], [493,542], [588,536], # 2
                          [397,565], [497,560], [595,554]  # 4
                          ])
    rpoints = np.float32([#      1          5          9
                          [411,498], [499,502], [589,505], # 1
                          [399,514], [488,518], [580,521], # 3
                          [385,530], [476,535], [571,539]  # 5
                          ])

    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ])
    print("Corners without origin shift:")
    print("{}, {}, {},".format(p3D[0], p3D[1], p3D[2]))
    print("{}, {}, {},".format(p3D[3], p3D[4], p3D[5]))
    print("{}, {}, {},".format(p3D[6], p3D[7], p3D[8]))
    npts  = np.float32([shiftOrigin(R, T, p, None) for p in p3D])
    print("Corners after origin shift:")
    print("{}, {}, {},".format(npts[0], npts[1], npts[2]))
    print("{}, {}, {},".format(npts[3], npts[4], npts[5]))
    print("{}, {}, {},".format(npts[6], npts[7], npts[8]))

    ##-#######################################################################################
    ## Test 3: Six balls on the ground
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 6 balls manually measured
    ##  World coordinates: distance between balls
    print("\n\n-------------------------------------------------------------------")
    print(" Test 3 : six balls on the ground")
    lpoints = np.float32([[415.,533.], [641.,519.], [796.,510.],    ## Left  p00, p01, p11
                          [425.,626.], [684.,612.], [860.,598.]])   ## Left  p10, p11, p12
    rpoints = np.float32([[424.,502.], [642.,510.], [801.,517.],    ## Right p00, p01, p11
                          [355.,588.], [599.,604.], [783.,611.]])   ## Right p10, p11, p12

    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ])
    print("Points before origin shift:")
    print("{}, {}, {},".format(p3D[0], p3D[1], p3D[2]))
    print("{}, {}, {},".format(p3D[3], p3D[4], p3D[5]))
    npts  = np.float32([shiftOrigin(R, T, p, None) for p in p3D])
    print("Points after origin shift:")
    print("{}, {}, {},".format(npts[0], npts[1], npts[2]))
    print("{}, {}, {},".format(npts[3], npts[4], npts[5]))

    ##-#######################################################################################
    ## Ball distances
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
    print("Test 3B: Relative distance measurements between balls\n")
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

    ##-#######################################################################################
    ## Test 2: 2 balls on the ground. New images
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 2 balls
    ##  World coordinates: (X,Y,Z) of both balls
    ##-#######################################################################################
    ## First picture
    ## Left point: [373, 536]
    ## Right point: [383, 499]
    ## Left undistorted point: [[[322.16092 513.7557 ]]]
    ## Right undistorted point: [[[423.06592 510.04587]]]
    ## point3D: [-2.2075872e+00 6.7742485e+01 8.3029316e+03]
    ## 11/05/19 12:43:42.598 INFO >> Real Word coordinates: X:-2 Y:68 Z:8303
    ## /mnt/R931GB/basler_test4/LEFT/br_img_1557575000.png
    ## 
    ## ==========
    ## fourth picture:
    ## Left point: [642, 522]
    ## Right point: [641, 510]
    ## Left undistorted point: [[[592.833 515.02686]]]
    ## Right undistorted point: [[[694.8344 509.63968]]]
    ## point3D: [1301.4044 69.96541 8320.072 ]
    ## 11/05/19 12:47:22.339 INFO >> Real World coordinates: X:1301 Y:70 Z:8320
    ## 
    ## /mnt/R931GB/basler_test4/LEFT/br_img_1557575252.png
    ##
    ## L Coordinates :
    lpoints = np.array([[373., 536.], [642., 522.], [479., 238.]],   ## Left  p00, p10
                        dtype=np.float32) 
    ## R Coordinate :
    rpoints = np.array([[383., 499.], [641., 510.], [306., 217.]],   ## Right p00, p10
                        dtype=np.float32) 

    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ])
    print("\n\nPoints before origin shift:")
    print(p3D) 

    npts  = np.float32([shiftOrigin(R, T, p, None) for p in p3D])
    print("Points after origin shift:")
    print(npts)
    print("Last output is expected to be X: 843   Y: 1843   Z: 5981. HOWEVER, ") 
    print("                                  ^--X measurement doesn't seem correct here")
    print("On the other hand, output generated by the script (300) seems correct)")
    # Last expected output is supposed to be 
    # X: 843    Y: 1843    Z: 5981. X doesn't seem correct here (output generated seems correct)
    #     ^---this seems to be measured from the cage wall. Our reference X=0 is left cage wall
    ## Output:      [  561.985718  1882.111816  6029.301758]



#EOF