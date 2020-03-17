# depthFinderPy
Find world 3D coordinates from a pair of image correspondence from left and right views

## Find depth in real world co-ordinates
Use [depth_finder.py](depth_finder.py) for this purpose. There are also some other example scripts (depth_tester*.py) that demonstrate the usage. Needs camera pose and camera calibration data. Input is left and right (x,y) image co-ordinates of the object for which real-world (X,Y,Z) co-ordinates are to be computed.

When you compute depth from stereo left and right images, the co-ordinates that you get are in the camera co-ordinates system. This may work for most applications where output is used on a computer monitor (which would match camera co-ordinate system). However, to compute real world co-ordinates -- for example, to treat the ground floor as y=0 plane, and a specific point on floor as the real-world origin, _and_ compute all co-ordinates accordingly -- we need to compute camera pose and use that to compute real world co-ordinates from camera co-ordinate system.


## Pose estimation
In order to compute the real world co-ordinates, we need to know the ground plane (or whichever plane you want to use as ground plane). This requires a special calibration known as pose estimation. Use [pose_finder.py](pose_finder.py) utility to compute the pose data required by depth_finder. It computes accurate camera pose using 3D points for a given pair of left and right images of a calibration target.

## Stereo calibration
Use [stereo_calibration.py](stereo_calibration.py) to compute intrinsics and extrinsics needed by the main depth_finder script. This is 
