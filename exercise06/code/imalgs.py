import cv2 
import numpy as np
import matplotlib.pyplot as plt 

def txt2array(filename, sep='\n'):
    txt = open(filename).read() 
    return np.fromstring(txt, dtype=float, sep=sep)

def imread(filename):
    return cv2.imread(filename) 

def imshow(img, name='img', size=(15,5), cmap='gray'): #'viridis'
    plt.subplots(figsize=size) 
    plt.imshow(img, cmap = cmap) 
    plt.title(name)
    plt.show() 

def imcompare(img1, img2, size=(30,10)):
    plt.figure(figsize=figsize)

    plt.subplot(1,3,2)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(1,3,3)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.show() 

def immatches(img1, img2, p1, p2, figsize=(30,10)):
    # Display matched points
    plt.figure(figsize=figsize)

    plt.subplot(1,3,2)
    plt.imshow(img1);
    plt.plot(p1[0,:], p1[1,:], 'ys');
    plt.title('Image 1')

    plt.subplot(1,3,3)
    plt.imshow(img2);
    plt.plot(p2[0,:], p2[1,:], 'ys');
    plt.title('Image 2')

    plt.show() 
