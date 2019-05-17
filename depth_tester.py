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




## A 9x6 chessboard object points with 115 mm squares
opoints = np.float32([
         [    0.000000,     0.000000,     0.000000],
         [  115.000000,     0.000000,     0.000000],
         [  230.000000,     0.000000,     0.000000],
         [  345.000000,     0.000000,     0.000000],
         [  460.000000,     0.000000,     0.000000],
         [  575.000000,     0.000000,     0.000000],
         [  690.000000,     0.000000,     0.000000],
         [  805.000000,     0.000000,     0.000000],
         [  920.000000,     0.000000,     0.000000],
         [    0.000000,   115.000000,     0.000000],
         [  115.000000,   115.000000,     0.000000],
         [  230.000000,   115.000000,     0.000000],
         [  345.000000,   115.000000,     0.000000],
         [  460.000000,   115.000000,     0.000000],
         [  575.000000,   115.000000,     0.000000],
         [  690.000000,   115.000000,     0.000000],
         [  805.000000,   115.000000,     0.000000],
         [  920.000000,   115.000000,     0.000000],
         [    0.000000,   230.000000,     0.000000],
         [  115.000000,   230.000000,     0.000000],
         [  230.000000,   230.000000,     0.000000],
         [  345.000000,   230.000000,     0.000000],
         [  460.000000,   230.000000,     0.000000],
         [  575.000000,   230.000000,     0.000000],
         [  690.000000,   230.000000,     0.000000],
         [  805.000000,   230.000000,     0.000000],
         [  920.000000,   230.000000,     0.000000],
         [    0.000000,   345.000000,     0.000000],
         [  115.000000,   345.000000,     0.000000],
         [  230.000000,   345.000000,     0.000000],
         [  345.000000,   345.000000,     0.000000],
         [  460.000000,   345.000000,     0.000000],
         [  575.000000,   345.000000,     0.000000],
         [  690.000000,   345.000000,     0.000000],
         [  805.000000,   345.000000,     0.000000],
         [  920.000000,   345.000000,     0.000000],
         [    0.000000,   460.000000,     0.000000],
         [  115.000000,   460.000000,     0.000000],
         [  230.000000,   460.000000,     0.000000],
         [  345.000000,   460.000000,     0.000000],
         [  460.000000,   460.000000,     0.000000],
         [  575.000000,   460.000000,     0.000000],
         [  690.000000,   460.000000,     0.000000],
         [  805.000000,   460.000000,     0.000000],
         [  920.000000,   460.000000,     0.000000],
         [    0.000000,   575.000000,     0.000000],
         [  115.000000,   575.000000,     0.000000],
         [  230.000000,   575.000000,     0.000000],
         [  345.000000,   575.000000,     0.000000],
         [  460.000000,   575.000000,     0.000000],
         [  575.000000,   575.000000,     0.000000],
         [  690.000000,   575.000000,     0.000000],
         [  805.000000,   575.000000,     0.000000],
         [  920.000000,   575.000000,     0.000000]
         ])





def print3D(pts):
    # [r*9 + {0|4|8}]
    for r in [0,3,5]:
        pstr = ""
        for c in [0,4,8]:
            pstr += "{}, ".format(pts[r*9+c] - opoints[r*9+c])
        print(pstr)
            

if __name__ == '__main__':
    fmt = lambda x: "%12.4f" % x
    np.set_printoptions(formatter={'float_kind':fmt})
    print("\n-------------------------------------------------------------------")
    print("All results are in mm scale")
    print("All manual measurements are from top left cage corner")
    print("-------------------------------------------------------------------\n")
    intrinsics = "data/calib/intrinsics.yml"
    extrinsics = "data/calib/extrinsics.yml"

    ## Instantiate depthFinder; This will carry out all the required one-time setup including rectification
    df = depth_finder.depthFinder(intrinsics, extrinsics)


    ##-#######################################################################################
    ## Test 1: Chessboard flat on the ground.
    ## Available measurements: 
    ##  Image coordinates: (x,y) on left+right images of 9 corners manually measured
    ##  World coordinates: (X,Y,Z) of top left corner. Chessboard squares size = 115 mm
    ##-#######################################################################################
#     lpoints = np.float32([#      1          5          9
#                           [396,531], [489,526], [583,520], # 1
#                           [396,548], [493,542], [588,536], # 3
#                           [397,565], [497,560], [595,554]  # 5
#                           ])
#     rpoints = np.float32([#      1          5          9
#                           [411,498], [499,502], [589,505], # 1
#                           [399,514], [488,518], [580,521], # 3
#                           [385,530], [476,535], [571,539]  # 5
#                           ])
    lpoints = np.float32([
                         [  387.343414,   381.459198],
                         [  410.985657,   379.676788],
                         [  435.215637,   378.934570],
                         [  458.660950,   377.410614],
                         [  482.471222,   376.351288],
                         [  505.756927,   374.868744],
                         [  529.261414,   373.749054],
                         [  553.467712,   372.813416],
                         [  577.255310,   371.747620],
                         [  388.331512,   404.146088],
                         [  412.109100,   403.194641],
                         [  435.881989,   401.678040],
                         [  459.408234,   400.468811],
                         [  483.213440,   399.170563],
                         [  507.049133,   397.958527],
                         [  530.490601,   396.900818],
                         [  553.972961,   395.693146],
                         [  577.462036,   394.691406],
                         [  389.548553,   427.373657],
                         [  413.226135,   425.900757],
                         [  437.270691,   424.857391],
                         [  460.294922,   423.235168],
                         [  483.948730,   422.135742],
                         [  507.399658,   420.923889],
                         [  530.912231,   419.717224],
                         [  554.658081,   418.454468],
                         [  577.693604,   417.421783],
                         [  390.664948,   450.072357],
                         [  414.090729,   449.184906],
                         [  437.511810,   447.537415],
                         [  461.340057,   446.312378],
                         [  484.613922,   445.105408],
                         [  508.145172,   443.687988],
                         [  531.328125,   442.413940],
                         [  555.059082,   441.228424],
                         [  578.130127,   439.983124],
                         [  392.157257,   473.114502],
                         [  415.268768,   471.435944],
                         [  439.506439,   470.369080],
                         [  462.251862,   469.058868],
                         [  485.637726,   467.553375],
                         [  508.962158,   466.131409],
                         [  531.969910,   465.150238],
                         [  555.436401,   463.697174],
                         [  578.875061,   462.571991],
                         [  393.349762,   495.357269],
                         [  416.678375,   494.007904],
                         [  440.329987,   492.720947],
                         [  463.356506,   491.464722],
                         [  486.413269,   489.859497],
                         [  509.571198,   488.713409],
                         [  532.709839,   487.346680],
                         [  556.234192,   485.833527],
                         [  579.240417,   484.799500]
        ])    
    rpoints = np.float32([
                         [  409.843964,   353.722687],
                         [  431.418030,   353.679169],
                         [  453.960022,   354.797852],
                         [  476.226624,   355.123016],
                         [  498.832458,   355.648041],
                         [  521.262939,   355.498688],
                         [  543.929871,   356.935883],
                         [  566.887512,   357.473267],
                         [  589.987976,   358.254852],
                         [  410.869965,   375.541565],
                         [  432.997742,   376.421783],
                         [  454.850708,   376.816833],
                         [  477.237549,   377.465424],
                         [  499.458801,   377.920837],
                         [  522.203552,   378.922729],
                         [  544.910461,   379.378510],
                         [  567.497620,   380.073944],
                         [  590.521301,   380.909576],
                         [  412.093567,   398.005280],
                         [  433.492371,   398.709534],
                         [  455.812775,   399.481812],
                         [  477.768005,   399.580841],
                         [  500.297546,   400.632324],
                         [  522.749023,   401.161224],
                         [  545.316711,   401.966492],
                         [  568.016113,   402.661133],
                         [  591.028137,   403.509674],
                         [  413.120697,   419.599030],
                         [  434.663452,   420.958557],
                         [  456.320557,   421.136536],
                         [  478.827393,   422.040985],
                         [  501.091553,   422.740692],
                         [  523.352478,   423.501923],
                         [  545.768433,   424.204285],
                         [  568.654419,   425.029846],
                         [  591.329346,   425.646332],
                         [  414.253845,   441.902557],
                         [  435.458130,   442.620056],
                         [  457.733276,   443.427002],
                         [  479.432556,   444.066742],
                         [  501.839325,   445.049927],
                         [  524.073486,   445.487457],
                         [  546.732178,   446.385986],
                         [  569.144409,   447.362122],
                         [  591.556885,   448.220703],
                         [  415.367584,   463.587067],
                         [  437.025543,   464.531250],
                         [  458.897430,   465.269073],
                         [  480.922699,   466.065735],
                         [  502.544891,   466.729584],
                         [  525.112183,   467.652863],
                         [  547.398743,   468.590851],
                         [  569.734741,   469.367340],
                         [  592.404846,   470.197205],
                         ])    
    p3D = np.float32([df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ])
    print("Corners without origin shift:")
    print3D(p3D)
    cv.estimateAffine3D(opoints, p3d)

    sys.exit()

    #rvec = np.float32([   -1.21569432, -0.07247885,   -0.12604635]) ##    Rotation vector; chessboard flat on ground
    #tvec = np.float32([-1042.88191484, 48.05017955, 8394.73175675]) ## Translation vector; chessboard flat on ground
    
    # Chessboard perpendicular to ground; original rvec/tvec
    rvec = np.float32([    [0.29818819],    [-0.13998303],    [-0.03777255]])    
    #tvec = np.float32([-1069.72603459,  -676.62133635,  8275.26055723])
    ## Chessboard pattern perpendicular to ground. Fine tuned rvec/tvec
    #rvec = np.float32([[    0.0],    [-0.0],    [-0.13777255]])
    tvec = np.float32([[-1069.72603459],  [-676.62133635],  [8275.26055723]])

    Rt, _ = cv.Rodrigues(rvec)
    #pref = np.float32([111, 54, 8418]) # top left chessboard corner
    ## Move origin (0,0.0) to the top left corner on chessboard
    np3D = [np.dot((p3D[i]-p3D[0]), Rt) for i in range(len(p3D)) ]
    print("Corners after origin shift: np.dot((p3D[i]-p3D[0]), Rt)")
    print3D(np3D)
    
    p3d = np.expand_dims(p3D, axis=2)
    np3D = [np.dot(Rt.T, (p3d[i]-p3d[0])).squeeze() for i in range(len(p3d)) ]
    print("Corners after origin shift: np.dot(Rt, (p3d[i]-p3d[0]))")
    print3D(np3D)
    np3D = [(np.dot(Rt, p3d[i]) - tvec).squeeze() for i in range(len(p3d)) ]
    print("Corners after origin shift: (np.dot(Rt, p3d[i]) - tvec)")
    print3D(np3D)

    sys.exit()
    ## Result with
    ## No origin transfer
    ## [  111    54  8418], [  566    55  8435], [ 1026    51  8420],
    ## [  104   134  8239], [  566   132  8216], [ 1022   128  8221],
    ## [  101   211  8021], [  564   213  7994], [ 1023   211  8004],
    #
    ## Result with
    ## rvec = np.float32([-1.21569432, -0.07247885, -0.12604635])      ## Rotation vector; chessboard flat on ground
    ## pref = p3D[0] = [111, 54, 8418]
    ## np3D = [np.dot((p3D[i]-p3D[0]), Rt) for i in range(len(p3D)) ]
    ## [    0     0     0], [  453    46    12], [  907   121     9],
    ## [  -34   192    13], [  421   276     9], [  874   332    12],
    ## [  -68   421     9], [  388   510     7], [  843   562    15],
    #
    ## Result with
    ## rvec = np.float32([-1.21569432, -0.07247885, -0.12604635])      ## Rotation vector; chessboard flat on ground
    ## tvec = np.float32([-1042.88191484, 48.05017955, 8394.73175675]) ## Translation vector; chessboard flat on ground
    ## pref = tvec
    ## np3D = [np.dot((p3D[i]-tvec), Rt) for i in range(len(p3D)) ]
    ## [ 1146   137    27], [ 1598   183    40], [ 2053   258    36],
    ## [ 1112   329    41], [ 1567   414    36], [ 2019   469    39],
    ## [ 1078   558    36], [ 1533   647    35], [ 1989   699    42],










    ##-#######################################################################################
    ## Test 1 : Ball distances
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
    print("                   Compare these visually to the above (X,Y,Z) coordinates -- they match\n")
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

    p3D = [df.get3D(lpoints[i], rpoints[i]) for i in range(lpoints.shape[0]) ]
    ## Move origin (0,0.0) to the first row center ball (p01) 
    np3D = [np.dot((p3D[i]-p3D[1]), Rt) for i in range(len(p3D)) ]
    
    print('''
##-#######################################################################################
## Note 2: Two additional measurements below now behave as you were expecting: 
##-#######################################################################################
    ''')

    for i in range(len(p3D)):
        print("Point {}: {}".format(i, np3D[i]))
    print("In above output both Y and Z should be close to zero. Verify. Only X should change")
    


#EOF