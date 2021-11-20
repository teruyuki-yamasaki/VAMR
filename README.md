# VAMR - Vision Algorithms for Mobile Robotics

## About
[Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html) is a course of the ETH Zurich Master in Robotics, Systems and Control, lectured by Prof. Dr. Davide Scaramuzza, the director of the Robotics and Perception Group. He introduces us to the fundamental computer vision algorithms used in mobile robotics:

This course contains the exercise parts, in which students can gain hands-on expericences of implementing the computer vision algotithms provided during the lectures in order to have a deeper understanding of each topic. In this repository, I place what I have implemented for each exercise with some descriptions. While the main language in the exercises is MATLAB, I implement the algorithms in Python.

## Exercises
### 01. [Augmented reality wireframe cube](https://github.com/teruyuki-yamasaki/VA4MR/tree/main/exercise01)

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise01/results/cube_dots_distorted.png"/>

### 02. [the Perspective-n-Points (PnP) problem](https://github.com/teruyuki-yamasaki/VA4MR/tree/main/exercise02)
To estimate the camera pose for each image in a given dataset, using corresponding pixel and world coordinates of the reference points provided for each image, as well as the camera matrix: the Direct Linear Transform (DLT) algorithm

### 03. [Harris detector + descriptor + matching](https://github.com/teruyuki-yamasaki/VA4MR/tree/main/exercise03)
To implement Harris corner detection and key point tracking: First evaluate the Harris score for each pixel of the input image, then select key points based on the Harris scores, and finally match descriptors in order to find feature correnspondences between frames. 

### 04. [SIFT detector + descriptor + matching](https://github.com/teruyuki-yamasaki/VAMR/tree/main/exercise04)

### 05. [Stereo Dense Reconstruction](https://github.com/teruyuki-yamasaki/VAMR/tree/main/exercise05)

### 06. [Two-View Geometry](https://github.com/teruyuki-yamasaki/VAMR/tree/main/exercise06)

### 07. [From image to localization](https://github.com/teruyuki-yamasaki/VAMR/tree/main/exercise07)
