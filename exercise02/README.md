# Exercise 02 : the Perspective-n-Point Problem

To estimate the camera pose for each image in a given dataset, using corresponding pixel and world coordinates of the reference points provided for each image, as well as the camera matrix: the Direct Linear Transform (DLT) algorithm

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise02/results/pnp.png"/>


dataset:
```
K (3,3) = 
 [[420.506712   0.       355.208298]
 [  0.       420.61094  250.336787]
 [  0.         0.         1.      ]]

 p_W_corners (3, 12) =  
 [[  0.    10.4    0.    10.4    0.    10.4    0.    10.4   19.4   19.4
   19.4   19.4 ]
 [  8.     8.    21.5   21.5   26.6   26.6   26.6   26.6   14.6   14.6
   25.    25.  ]
 [  0.     0.     0.     0.   -10.95 -10.95 -24.45 -24.45 -10.95 -24.45
  -10.95 -24.45]]

 detected_corners (210, 2, 12) =  
 [[[186.5 264.5 218.5 ... 495.5 396.5 496.5]
  [187.5 166.5 304.5 ... 211.5 280.5 300.5]]

 [[186.5 264.5 217.5 ... 495.5 396.5 496.5]
  [186.5 166.5 303.5 ... 211.5 279.5 300.5]]

 [[187.5 265.5 218.5 ... 495.5 397.5 496.5]
  [186.5 166.5 303.5 ... 211.5 279.5 300.5]]

 ...

 [[220.5 293.5 252.5 ... 543.5 433.5 547.5]
  [216.5 194.5 342.5 ... 225.5 306.5 319.5]]

 [[214.5 289.5 248.5 ... 537.5 429.5 542.5]
  [226.5 203.5 352.5 ... 233.5 313.5 327.5]]

 [[208.5 283.5 242.5 ... 531.5 424.5 537.5]
  [236.5 212.5 362.5 ... 243.5 322.5 336.5]]]

 filenames = 
../data/images_undistorted/img_0001.jpg
../data/images_undistorted/img_0002.jpg
../data/images_undistorted/img_0003.jpg
...
```


homogenous coordinates:
```
 Pwh = 
 [[  0.    10.4    0.    10.4    0.    10.4    0.    10.4   19.4   19.4
   19.4   19.4 ]
 [  8.     8.    21.5   21.5   26.6   26.6   26.6   26.6   14.6   14.6
   25.    25.  ]
 [  0.     0.     0.     0.   -10.95 -10.95 -24.45 -24.45 -10.95 -24.45
  -10.95 -24.45]
 [  1.     1.     1.     1.     1.     1.     1.     1.     1.     1.
    1.     1.  ]]

 ph = 
 [[186.5 264.5 218.5 285.5 292.5 356.5 406.5 461.5 388.5 495.5 396.5 496.5]
 [187.5 166.5 304.5 270.5 369.5 322.5 418.5 353.5 204.5 211.5 280.5 300.5]
 [  1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1. ]]
```


comparison of the detected corners and the reprojected corners:
```
 q = 
 [[186.5 264.5 218.5 285.5 292.5 356.5 406.5 461.5 388.5 495.5 396.5 496.5]
 [187.5 166.5 304.5 270.5 369.5 322.5 418.5 353.5 204.5 211.5 280.5 300.5]]

 p = 
 [[186.22043116 264.38885286 218.16156306 285.41466996 292.15334011
  355.29379766 404.84774683 460.83775393 387.57288526 494.05378042
  395.56263438 495.33729506]
 [187.07076877 165.90760087 303.69275866 268.82378274 369.87951844
  321.85023944 419.36957946 353.0450511  203.85727324 211.07376434
  279.56733973 299.53658842]]

```
