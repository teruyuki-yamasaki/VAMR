# DoG


[code](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/code/main.py)

### dataset
used images:

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/img1img2.png"/>

```
*****Img1*****
type: <class 'numpy.ndarray'>
size: (3024, 4032)

*****Img2*****
type: <class 'numpy.ndarray'>
size: (3024, 4032)
```

### image gradient 
comparison ofr algorithms for computing the image gradients:
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/gradient.png"/>
```
--------------------test Image Gradient--------------------
the sum of gradient filter components = 0
alg=-1: 0.003129
shape = (15, 15, 2)
alg=0: 0.002807
shape = (15, 15, 2)
alg=1: 0.002298
shape = (13, 13, 2)
alg=2: 0.000520
shape = (13, 13, 2)
alg=3: 0.000132
shape = (15, 15, 2)
```

### Gaussian Filter 
comparison of algorithms for perfoming the Gaussian filtering:
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/GaussFilter3.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/gaussed.png"/>
```
--------------------test Gaussian Filter--------------------
the sum of the gaussian filter components = 0.9999999999999998
alg=0: 0.001472
shape= (15, 15)
alg=1: 0.000550
shape= (9, 9)
alg=2: 0.000918
shape= (9, 9)
alg=3: 0.000116
shape= (15, 15)
```

### SHIFT detectors:
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/img1part.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise04/results/img2part.png"/>
