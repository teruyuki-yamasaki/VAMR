import numpy as np 
import time 
from constructStructureTensor import * 

def run_test_structurertensor():
    t0 = time.time()
    M_norm = StructureTensorM(dI, patch_radius=2)
    t1 = time.time() 
    print('{:.2f}'.format(t1-t0))
    imshow(M_norm[:,:,0]/np.linalg.norm(M_norm[:,:,0])*255, 'Ix**2')
 
if __name__=="__main__":
    run_test_structuretensor() 
