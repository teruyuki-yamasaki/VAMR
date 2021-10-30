from google.colab import drive
drive.mount('/content/drive')
%cd '/content/drive/MyDrive/ETH/VA4MR/exercise4/'

# Set parameters 
S = 3 # the number of scales per octave 
O = 5 # the number of octaves 
sigma0 = 1.6 # base sigma 

import cv2 
import matplotlib.pyplot as plt 

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 

def imshow(img, name='img', size=(30,10)):
    plt.subplots(figsize=size) 
    plt.imshow(img) 
    plt.title(name)
    plt.show() 

def showTwoImages(img1, img2, size=(30,20)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ax1.imshow(img1) 
    ax1.set_title('img_1')
    ax2.imshow(img2) 
    ax2.set_title('img_2')
    plt.show() 

img1 = imread('images/img_1.jpg')
img2 = imread('images/img_2.jpg')

showTwoImages(img1, img2) 

import numpy as np

def padarray(img, r):
    Img = np.zeros((len(img)+2*r, len(img[0])+2*r))
    Img[r:-r, r:-r][:] = img[:] 
    return Img

def GaussianFilter(sigma, r): 
    x = np.arange(2*r+1)
    y = np.arange(2*r+1) 
    xx, yy = np.meshgrid(x, y)
    return np.exp(- ((xx - r)**2 + (yy - r)**2) / sigma**2) / (2 * np.pi * sigma**2) 

def GaussianBlur(img, sigma, r=1):
    G = GaussianFilter(sigma, r)
    img_pad = padarray(img, r) 

    #img_G = cv2.filter2D(img, -1, G)

    img_G = np.zeros_like(img)
    for v in range(len(img)):
        for u in range(len(img[0])):
            img_G[v, u] = np.sum(img_pad[v:v+2*r+1, u:u+2*r+1] * G)
    
    return img_G
   
img = np.random.randint(256, size=(15,15))
imshow(GaussianFilter(sigma=1.6, r=1))
showTwoImages(img, GaussianBlur(img, sigma=2, r=1)) 
showTwoImages(img1, GaussianBlur(img1, sigma=1.6))

def blurImages(img, S):
    s_list = list(i for i in range(-1, S+2))

    imgG_list = np.zeros((len(s_list), img.shape[0], img.shape[1])) 
    for i, s in enumerate(s_list):
        sigma = 2 ** (s/S) * sigma0
        imgG_list[i] = GaussianBlur(img, sigma) 
        print(str(s) + ' is done')
    
    return imgG_list 

def getDoG(imgG_list):
    L, H, W = imgG_list.shape 

    DoG = np.zeros((L-1, H, W)) 

    for i in range(L-1):
        DoG[i] = imgG_list[i+1] - imgG_list[i] 

    return DoG

def downSample(img, octave):
    return img[::2**octave, ::2**octave] 

def getDoGoctaves(img, S, O):
    octaves = list(i for i in range(O+2))
    dogs = dict() 

    for o in octaves:
        imgo = downSample(img, o) 
        imgoG = blurImages(imgo, S) 
        DoGo = getDoG(imgoG)
        dogs[o] = DoGo
        print('*'*5 + "octave" + str(o) + ' is finished')
    
    return dogs
