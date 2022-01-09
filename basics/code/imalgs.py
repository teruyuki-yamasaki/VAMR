import numpy as np 
import cv2 


def impad(img,r=1, mode='constant'): 
    return  np.pad(img, pad_width=r, mode=mode)
