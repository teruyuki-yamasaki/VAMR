import numpy as np 
import cv2 
from scipy.ndimage.filters import gaussian_filter,convolve
import matplotlib.pyplot as plt
import time

def imgrad(img, filter='Sobel', alg=0):
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) if filter=='Sobel' \
        else np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Prewitt 

    if alg==0:
        dI = np.zeros((2, img.shape[0], img.shape[1])) 
        dI[0] = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        dI[1] = cv2.filter2D(img, ddepth=-1, kernel=kernel.T)
    
    elif alg==1:
        img = img.astype(float) 
        dI = np.zeros((2, img.shape[0], img.shape[1])) 
        dI[0] = convolve(img, kernel) 
        dI[1] = convolve(img, kernel.T) 
    
    elif alg==2:
        height, width = img.shape 
        r = kernel.shape[0]//2 
        patches = np.zeros(((2*r+1)**2, height+2*r, width+2*r), dtype=img.dtype)
        img = impad(img, [r,r]) 
        for y in range(2*r+1):
            for x in range(2*r+1):
                patches[(2*r+1)*y+x] = np.roll(img, -(width*y+x)).flatten().reshape(height+2*r, width+2*r) 
        patches = patches[:,r:-r, r:-r]
        dI = np.zeros((2, height, width)) 
        dI[0] = np.sum(patches.transpose(1,2,0)*kernel.flatten(), axis=-1) 
        dI[1] = np.sum(patches.transpose(1,2,0)*kernel.T.flatten(), axis=-1) 

    return dI
  
 
def impad(img, pad=[1, 1]):
    if pad[0]==0:
        return img
    else:
        img_pad = np.zeros((img.shape[0]+2*pad[0], img.shape[1]+2*pad[1]), dtype=img.dtype) 
        img_pad[pad[0]:-pad[0],pad[1]:-pad[1]] = img
    return img_pad 
  

def imshow(img, name="img"):
    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break
    cv2.destroyAllWindows()

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
 

num = 20 
def run_test_imgrad(filename):
    '''
        test imgrad 
    '''
    print('*'*num + 'test imgrad' + '*'*num) 
    print('with {}'.format(filename)) 

    # image 
    img = imread(filename)
    print(f'\n img{img.shape, img.dtype} = \n', img)
    imshow(img)

    # compare algorithms 
    for i in range(3):
        t0 = time.time()
        dI = imgrad(img, alg=i) 
        t1 = time.time()
        print(f'\n alg = {i} took {t1-t0:.4f} sec')

        IxIy = np.vstack((dI[0], dI[1]))
        imshow(IxIy, f'Ix and Iy: alg={i}')  

if __name__=="__main__":
    run_test_imgrad('../data/images/KTTI0.png')
    run_test_imgrad('../data/images/CircleLineRect.png') 
