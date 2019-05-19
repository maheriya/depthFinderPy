#!/bin/csh -f

## Generate pose.yml file for each chessboard image pair
## A. Default (perpendicular chessboard at 8231)
set datadir = "./data/calib"
set imgdir = "./data/images"

##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-perpendicular.png \
##  --right-image $imgdir/right-chessboard-perpendicular.png \
##  --offset 2.000000 -735.400024 8231
##cp $datadir/pose.yml $datadir/pose_90degree_Z8009.yml

##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-flat-5300-1.png \
##  --right-image $imgdir/right-chessboard-flat-5300-1.png \
##  --offset 310 -40 5300
##cp $datadir/pose.yml $datadir/pose_Z5300_1.yml


python pose_finder.py \
  --left-image $imgdir/left-chessboard-flat-5300-2.png \
  --right-image $imgdir/right-chessboard-flat-5300-2.png \
  --offset 1040 -40 5300
cp $datadir/pose.yml $datadir/pose_Z5300_2.yml


##python pose_finder.py \
##  --left-image $imgdir/left-chessboard-flat-5300-3.png \
##  --right-image $imgdir/right-chessboard-flat-5300-3.png \
##  --offset 2000 -40 5300
##cp $datadir/pose.yml $datadir/pose_Z5300_3.yml

echo "Done"

