# Exercise 05 : Stereo Dense Reconstruction 


## Outline
The goal of this exercise is to reconcstruct a 3D scene using dense epipolar matching ([sample video](https://www.youtube.com/watch?v=cyPFR61uuHA)). 

- determine pixel disparity between a pair of left and right stereo frame using SSD matching on a disparity range
- apply some of heuristics to remove outliers
- backproject the matched pixels and triangulate the corresponding 3D point
- pose information of the frames to accumulate a global point cloud and visualize it 

