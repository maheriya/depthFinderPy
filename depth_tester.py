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

## A 9x6 chessboard object points with 115 mm squares
def getObjPoints(w, h, squareSize, flat=False):
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp * squareSize
    if flat: ## If the target is parallel (or flat) to the ground
        objp[:,2] = -objp[:,1] ## Z = -Y [Invert for Z axis; Z reduces closer to camera]
        objp[:,1] = 0          ## Y = 0  [Flat on ground

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
    offset = np.float32([570, -40-(squareSize*6.+46.), npts[0][2]])
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




#EOF