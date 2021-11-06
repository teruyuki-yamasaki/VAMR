# VA4MR - [Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html)

## About
Vision Algorithms for Mobile Robotics is a course of the ETH Zurich Master in Robotics, Systems and Control, lectured by Prof. Dr. Davide Scaramuzza, the director of the Robotics and Perception Group. He introduces us to the fundamental computer vision algorithms used in mobile robotics:

This course contains the exercise parts, in which students can gain hands-on expericences of implementing the computer vision algotithms provided during the lectures in order to have a deeper understanding of each topic. In this repository, I place what I have implemented for each exercise with some descriptions. While the main language in the exercises is MATLAB, I write the codes in Python.

## Exercises
### 01. Augmented reality wireframe cube
To superimpose a virtual cube on a video of a checker board viewed from different positions and orientation: 
After determining corner positions of a virtual cube in a given three-dimensional world coordinate system that is associated with a physical planner grid, 
we calculate their corresponding pixel coordinates in each frame, using the camera's poses and orientations in the world coordinate system (extrinsic parameters) as well as the camera matrix (intrinsic parameters).

### 02. the Perspective-n-Points (PnP) problem
To estimate the camera pose for each image in a given dataset, using corresponding pixel and world coordinates of the reference points provided for each image, as well as the camera matrix: the Direct Linear Transform (DLT) algorithm

### 03. Harris detector + descriptor + matching
To implement Harris corner detection and key point tracking: First evaluate the Harris score for each pixel of the input image, then select key points based on the Harris scores, and finally match descriptors in order to find feature correnspondences between frames. 

### 04. SIFT detector + descriptor + matching

### 05. Stereo vision: rectification, epipolar matching, disparity, triangulation
