# Exercise 05 : Stereo Dense Reconstruction 


## Outline
The goal of this exercise is to reconcstruct a 3D scene using dense epipolar matching ([sample video](https://www.youtube.com/watch?v=cyPFR61uuHA)). 

- determine pixel disparity between a pair of left and right stereo frame using SSD matching on a disparity range
- apply some of heuristics to remove outliers
- backproject the matched pixels and triangulate the corresponding 3D point
- pose information of the frames to accumulate a global point cloud and visualize it 

We make use of the public [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise05/results/images_left_right.png"/>

## part 1: calculate pixel disparity 
SSDの計算([code](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise05/code/calc_ssd.py))

disparityの計算([code](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise05/code/disparity.py))

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise05/results/disparity_ssd.png"/>

## part 2: outlier removal 
Reject all amibuous matches. 
