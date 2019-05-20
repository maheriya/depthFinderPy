#!/bin/csh -f

## Generate pose.yml file for each chessboard image pair
## A. Default (perpendicular chessboard at 8231)
set datadir = "./data/calib"
set imgdir = "./data/images"

## Offset (due to chessboard patterns distance from board physical corners)

##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-perpendicular.png \
##  --right-image $imgdir/right-chessboard-perpendicular.png \
##  --offset 2.000000 -735.400024 8231
##cp $datadir/pose.yml $datadir/pose_90degree_Z8009.yml

## First : X:  610 Y: -40 Z 5140
##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-flat-5300-1.png \
##  --right-image $imgdir/right-chessboard-flat-5300-1.png \
##  --offset 610 -40 5300
##cp $datadir/pose.yml $datadir/pose_Z5300_1.yml

##Second: X: 1315 Y: -40 Z 5185
python pose_finder.py \
  --left-image $imgdir/left-chessboard-flat-5300-2.png \
  --right-image $imgdir/right-chessboard-flat-5300-2.png \
  --offset 1315 -40 5185
cp $datadir/pose.yml $datadir/pose_Z5300_2.yml


## Third : X: 2320 Y: -40 Z 5140
##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-flat-5300-3.png \
##  --right-image $imgdir/right-chessboard-flat-5300-3.png \
##  --offset 2320 -40 5140
##cp $datadir/pose.yml $datadir/pose_Z5300_3.yml

echo "Done"

