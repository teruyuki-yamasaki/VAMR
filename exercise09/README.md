# Bandle Adjustment 

## Input Data 
```
import os 
import numpy as np 
import matplotlib.pyplot as plt 

DIR_DATA = '../data' 

def main():
    hidden_state = loadtxt('hidden_state.txt')
    K = loadtxt('K') 
    observations = loadtxt('observations') 
    poses = loadtxt('poses') 

    npprint(hidden_state, 'hidden_state')  
    npprint(K, 'K')
    npprint(observations, 'observations')
    npprint(poses, 'poses')
 
def loadtxt(filename):
    if '.txt' not in filename: filename += '.txt' 
    return np.loadtxt(os.path.join(DIR_DATA, filename))

def npprint(array, name='array'):
    print()
    print(f'{name}{array.shape, array.dtype} = \n', array) 

if __name__=="__main__":
    main() 
>>
hidden_state((106176,), dtype('float64')) = 
 [-7.9228e-03 -1.1995e-03  1.2009e+00 ...  4.1298e+00  5.5609e-01
  2.8606e+01]

K((3, 3), dtype('float64')) = 
 [[718.856    0.     607.1928]
 [  0.     718.856  185.2157]
 [  0.       0.       1.    ]]
(Brains) yamasaki@MacBook-Pro-100 code % python main.py

hidden_state((106176,), dtype('float64')) = 
 [-7.9228e-03 -1.1995e-03  1.2009e+00 ...  4.1298e+00  5.5609e-01
  2.8606e+01]

K((3, 3), dtype('float64')) = 
 [[718.856    0.     607.1928]
 [  0.     718.856  185.2157]
 [  0.       0.       1.    ]]

observations((377154,), dtype('float64')) = 
 [2.5000e+02 3.4892e+04 2.5000e+01 ... 3.4890e+04 3.4891e+04 3.4892e+04]

poses((250, 12), dtype('float64')) = 
 [[ 9.999433e-01  2.586172e-03 -1.033094e-02 ...  5.797170e-03
   9.999299e-01  4.291335e+00]
 [ 9.999184e-01  3.088363e-03 -1.239599e-02 ...  6.961759e-03
   9.998991e-01  5.148987e+00]
 [ 9.998890e-01  3.586305e-03 -1.446384e-02 ...  8.129853e-03
   9.998627e-01  6.007777e+00]
 ...
 [ 9.748174e-01  3.407297e-02  2.203864e-01 ...  1.779877e-02
   9.750952e-01  1.188364e+02]
 [ 9.743646e-01  4.180173e-02  2.210574e-01 ...  1.759609e-02
   9.749028e-01  1.196659e+02]
 [ 9.734172e-01  4.884577e-02  2.237701e-01 ...  1.513714e-02
   9.743027e-01  1.205026e+02]]

```
