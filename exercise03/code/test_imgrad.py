import numpy as np 
import time 
from imgrad import imgrad 

def run_test_imgrad():
  t0 = time.time() 
  dI = imgrad(img, alg=0) 
  t1 = time.time() 
  print('{:.2f}'.format(t1-t0))
  imshow(dI[:,:,0], 'Ix') 

  t0 = time.time() 
  dI = imgrad(img, alg=1) 
  t1 = time.time() 
  print('{:.2f}'.format(t1-t0))
  imshow(dI[:,:,0], 'Ix') 
 
if __name__=='__main__':
  run_test_imgrad()
