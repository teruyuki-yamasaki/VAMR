import numpy as np
import cv2 
import matplotlib.pyplot as plt 
import time 

def main():
    ### 1. Load images ###
    # Show original images 
    img1 = imread('images/img_1.jpg')
    img2 = imread('images/img_2.jpg')

    if 0: # show info of loaded data 
        iminfo(img1, 'Img1')
        iminfo(img2, 'Img2') 
        imcompare(img1, img2) 
    
    if 1: # no need to use smaller data once you get faster algs 
        # Extract corresponding parts of the images because the original ones are too large to handle. 
        img1 = img1[1400:1600, 1400:1600]
        img2 = img2[1500:2000, 250:750]

        if 0:
            iminfo(img1, 'Img1')
            iminfo(img2, 'Img2') 
            imcompare(img1, img2) 

    if 0: # Show how Gaussian filters work
        run_test_gaussian()
    
    if 0:
        run_test_imgrad() 
    
    if 1:
        des1 = img2des(img1) 
        des2 = img2des(img2) 

        showkpts(img1, des1) 
        showkpts(img2, des2)   

def showkpts(img, descriptors):

    #print(len(descriptors))
    kpts = np.zeros((len(descriptors),2), dtype=int) # datatype ?? 

    print(len(kpts)) 

    cnt = 0 
    for des in descriptors:
        if des.octave == 0:
            kpts[cnt,0] = des.x 
            kpts[cnt,1] = des.y 
            cnt += 1

        else:
            pass 
        
    
    kpts = kpts[:cnt] 
    
    print(len(kpts)) 

    print(np.count_nonzero(kpts)) # 2634 

    img = impoints(img, kpts) 

    imshow(img)

def img2des(img, O=5, S=3, sigma0=1.6, r=2):
    gauss, diffs = octavatePyramids(img, O, S, sigma0, r) 

    D = list() 
    for octave in range(O):
        des = diffs2des(gauss[octave], diffs[octave], octave)
        D.extend(des)

    return D  

def diffs2des(gauss, diffs, octave, nonmax_suppression_radius=3): 
    diffs_shape = diffs.shape

    kpts = DoGpyramid2KeyPoints(diffs, nonmax_suppression_radius)

    descriptors = list() 
    #print(DoGpyramid.shape)
    for zyx in kpts: 
        z, y, x = zyx 
        #print(z, y, x) 

        if condition(x,y,z,diffs_shape): 
            img = gauss[z-1] 
            des = Descriptor(img, x, y, z, octave) 
            descriptors.append(des) 

    return descriptors

def condition(x, y, z, diffs_shape):
    depth, height, width = diffs_shape 
    return 0 < z and z < depth-1 and  7 <= x and x < height-8 and 7 <= y and y < width-8 

class Descriptor:
    def __init__(self, img, x, y, z, octave):
        self.x = x 
        self.y = y 
        self.z = z 
        self.octave = octave 
        self.des = self.descriptor(img, x, y) 
        self.normScaler = normScaler(sigma=1.5*16)
    
    def descriptor(self, img, x, y):
        patch = extract16x16Patch(img, x, y)  
        #print(img. shape, patch.shape)  
        Grt = imgradMagDir(patch)         
        Grt[:,:,0] = Grt[:,:,0] * normScaler(sigma=1.5*16) # ??

        des = np.zeros((4, 4, 8)) 
        for i in range(16):
            des[i//4, i%4, :], _ = weightedhistc(Grt[i//4:i//4+4, i%4:i%4+4,:])
        
        des = (des / np.sum(des)).flatten() # normalization

        return des 

def normScaler(sigma=1.5*16):
    x = np.arange(16)
    y = np.arange(16) 
    xx, yy = np.meshgrid(x, y) 
    kernel = np.exp(-((xx - 7)**2 + (yy - 7)**2) / sigma**2)
    return kernel / np.sum(kernel)
 

def extract16x16Patch(img, x, y):
    return img[x-7:x+9, y-7:y+9]

def weightedhistc(Grt):
    counts, bins = np.histogram((Grt[:,:,0]*Grt[:,:,1]).flatten(), bins=list(2*np.pi*i/8 - np.pi for i in range(9)))
    # edges = [-3.14159265 -2.35619449 -1.57079633 -0.78539816  0.  0.78539816 1.57079633  2.35619449  3.14159265]
    return counts, bins 

def showHistgram(X, a=0, b=1, N=15, ymax=5000):
    counts, bins = np.histogram(X.flatten(),bins=N)
    plt.hist(bins[:-1], bins[int(a*N):int(b*N)], weights=counts)
    plt.text(10,10,'max={}'.format(np.max(X))) 
    plt.ylim(-0.1,ymax)
    plt.show() 

def DoGpyramid2KeyPoints(diffs, C=1, nonmax_suppression_radius=5):
    ''' select key points given a DoG pyramid 

    Input
        - diffs: a Difference of Gaussian pyramid 
        - C: threshold for noise suppression 
        - nonmax_suppression_radius: 

    Output:
        - kpts: selected key points 
        - zxy: 
    '''
    diffs[diffs<=C] = 0 
    diffs = nonMaxSuppress3d(diffs, nonmax_suppression_radius) 
    kpts = selectKeyPoints3d(diffs)

    return kpts

def selectKeyPoints3d(X, r=1):
    if 0:
        kpts = np.zeros_like(X, dtype=bool) 
        zyx = [] 
        for s in range(X.shape[0]-2*r):
            for v in range(X.shape[1]-2*r):
                for u in range(X.shape[2]-2*r):
                    x = X[s:s+2*r+1, v:v+2*r+1, u:u+2*r+1] 
                    M = x[r,r,r]
                    x[r,r,r] = 0 
                    if M > np.max(x):
                        #print(s,v,u)
                        kpts[s+r, v+r, u+r] = True 
                        zyx.append([s+r,v+r,u+r]) 
    else:
        depth, height, width = X.shape  
        patches = np.zeros((depth, height, width, (2*r+1)**3))
        for z in range(2*r+1):
            for y in range(2*r+1):
                for x in range(2*r+1):
                    patch = np.roll(X.flatten(), -(width*height*z + width*y + x)).reshape(depth, height, width) 
                    patches[:,:,:, (2*r+1)**2*z + (2*r+1)*y + x] = patch 
                    #print((2*r+1)**2*z + (2*r+1)*y + x)
                        
        patches = patches[:-2*r, :-2*r, :-2*r, :]

        centers = patches[:,:,:,(2*r+1)**2*r + (2*r+1)*r + r + 1] 

        arounds = patches - centers.reshape(depth-2*r, height-2*r, width-2*r, 1) 

        maxes = np.max(arounds, axis=-1) # keepdims ? 

        centers[centers <= maxes] = 0

        boxes = np.zeros_like(X)

        boxes[r:-r,r:-r,r:-r] = centers    

        kpts = (boxes!=0)

        kptId = kpts.flatten() * np.arange(len(kpts.flatten()))
        kptId = kptId[kptId!=0] 

        zyx = np.zeros((len(kptId[1:]),3),dtype=np.uint32) 
        for i, id in enumerate(kptId[1:]):
            z = id // (height*width) 
            y = (id % (height*width)) // width
            x = (id % (height*width)) % width 

            zyx[i] = [z, y, x] 

    return zyx

def nonMaxSuppress2d(X, r=1):
    for v in range(len(X)-2*r):
        for u in range(len(X[0])-2*r):
            x = X[v:v+2*r+1, u:u+2*r+1] 
            M = np.max(x) 
            X[v:v+2*r+1, u:u+2*r+1] = (x== M)*M
    return X 

def nonMaxSuppress3d(X, r=1):
    if 1:
        for s in range(len(X)-2*r):
            for v in range(len(X[0])-2*r):
                for u in range(len(X[0][0])-2*r):
                    x = X[s:s+2*r+1, v:v+2*r+1, u:u+2*r+1] 
                    M = np.max(x) 
                    X[s:s+2*r+1, v:v+2*r+1, u:u+2*r+1] = (x==M)*M
    
    if 0:
        depth, height, width = X.shape  
        patches = np.zeros((depth, height, width, (2*r+1)**3))
        for z in range(2*r+1):
            for y in range(2*r+1):
                for x in range(2*r+1):
                    patch = np.roll(X.flatten(), -(width*height*z + width*y + x)).reshape(depth, height, width) 
                    patches[:,:,:, (2*r+1)**2*z + (2*r+1)*y + x] = patch 
                    print((2*r+1)**2*z + (2*r+1)*y + x)
                        
        patches = patches[:-2*r, :-2*r, :-2*r, :]

        maxes = np.max(patches, axis=-1, keepdims=True)  

        patches[patches < maxes] = 0 

        X[r:-r,r:-r,r:-r] = patches # the edges remains 
          
    return X 

def selectKeyPoints2d(X, r=1):
    if 1:
        kpts = np.zeros_like(X, dtype=bool) 
        for v in range(len(X)-2*r):
            for u in range(len(X[0])-2*r):
                x = X[v:v+2*r+1, u:u+2*r+1] 
                M = x[r,r]
                x[r,r] = 0 
                if M > np.max(x):
                    kpts[v+r, u+r] = True 
  
    return kpts 

def octavatePyramids(img, O=5, S=3, sigma0=1.6, r=2): 
    diffs = dict() 
    gauss = dict() 

    for octave in range(O):
        img_o = imdown(img, 2**octave)
        GP = GaussianPyramid(img_o, S, sigma0, r)   
        gauss[octave] = GP 
        DoGP = DoGpyramid(GP)
        diffs[octave] = DoGP 
        print('*'*5 + "octave " + str(octave) + ' is finished' + '*'*5)
    
    return gauss, diffs 

def DoGpyramid(Gaussians):
    '''compute a DoG pyramid given a Gaussian Pyramid 
        by cumputing difference between subsequent Gaussians 
    '''

    #print('Dogpyramid') 

    if 1:
        t0 = time.time()
        depth, height, width = Gaussians.shape 
        diffs = np.zeros((depth-1, height, width)) 
        for i in range(depth-1): 
            diffs[i] = np.absolute(Gaussians[i+1] - Gaussians[i])
        t1 = time.time()
        print('alg 1: %f ' % (t1-t0))
    
    if 1:
        t0 = time.time() 
        diffs = np.absolute(Gaussians[1:] - Gaussians[:-1]) 
        t1 = time.time() 
        print('alg 2: %f' % (t1 - t0)) 

    return diffs 

    '''
    alg 1: 0.425907 
    alg 2: 0.413119
    *****octave 0 is finished*****
    dog:
    alg 1: 0.076209 
    alg 2: 0.101482
    *****octave 1 is finished*****
    dog:
    alg 1: 0.024158 
    alg 2: 0.025666
    *****octave 2 is finished*****
    dog:
    alg 1: 0.003618 
    alg 2: 0.006779
    *****octave 3 is finished*****
    dog:
    alg 1: 0.000444 
    alg 2: 0.000643
    *****octave 4 is finished***** why little difference or even slower? 
    '''

def GaussianPyramid(img, S=3, sigma0=1.6, gauss_filter_radius=2):
    '''compute a Gaussian Pyramid given an image 
    '''

    height, width = img.shape 

    gaussians = np.zeros((S+3, height, width)) 

    for i, s in enumerate(list(range(-1, S+2))):
        sigma = 2 **(s / S) * sigma0 
        kernel = GaussianFilter(sigma, gauss_filter_radius) 
        gaussians[i] = imconv(img, kernel)  
    
    return gaussians


### imalgs 
def GradientFilter(method='Prewitt'):
    if method=='Prewitt':
        X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int8)
    
    elif method=='Sobel':
        X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)  
    
    return X

def GaussianFilter(sigma, r): 
    x = np.arange(2*r+1)
    y = np.arange(2*r+1) 
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(- ((xx - r)**2 + (yy - r)**2) / sigma**2) 
    return kernel / np.sum(kernel) # normalize it rathar than divide it by np.sqrt(2*np.pi*sigma**2) 

def imgradMagDir(img, method='Prewitt'): 
    op = GradientFilter(method)
    Gxy = imgrad(img, op) 

    ###TODO: check appropriate data type 
    #Grt = np.zeros_like(Gxy) 
    Grt = np.zeros(Gxy.shape, float)  
    Grt[:,:,0] = np.sqrt(Gxy[:,:,0]**2 + Gxy[:,:,1]**2) 
    Grt[:,:,1] = np.arctan2(Gxy[:,:,1], Gxy[:,:,0]) 
    return Grt

def imgrad(img, op, alg=3):  
    H, W = img.shape 

    opX, opY = op, op.T

    if alg==-1:
        dI = np.zeros((H, W, 2), dtype=int) 

        r = op.shape[0]//2 # 1
        img_pad = impad(img, [r,r])
        
        for v in range(H):
            for u in range(W): 
                I = img_pad[v:v+2*r+1, u:u+2*r+1]  
                dI[v, u, 0] = np.sum(I * opX)
                dI[v, u, 1] = np.sum(I * opY)
    
    else:
        Ix = imconv(img, opX, alg)  
        Iy = imconv(img, opY, alg)
        dI = np.zeros((Ix.shape[0], Iy.shape[1],2), dtype=int)  ## !! what int ?
        dI[:,:,0] = Ix 
        dI[:,:,1] = Iy 

    return dI 

def imconv(img, kernel, alg=3):
    height, width = img.shape
    r = kernel.shape[0] // 2 

    dtype = np.uint16 if np.min(kernel) >= 0 else np.int16 

    if alg==0: # 0.001630 / 73.178703
        ###TODO### make it more efficient
        img_pad = impad(img, [r,r]) 
        img_conv = np.zeros((height, width), dtype=dtype)    
        for y in range(height):
            for x in range(width):
                img_conv[y, x] = np.sum(img_pad[y:y+2*r+1, x:x+2*r+1].flatten() * kernel.flatten())
    
    elif alg==1: # edges are cut off 
        img_conv = np.zeros((height-2*r, width-2*r), dtype=dtype)  
        for y in range(height-2*r):
            for x in range(width-2*r):
                img_conv[y, x] = np.sum(img[y:y+2*r+1, x:x+2*r+1].flatten() * kernel.flatten())
    
    elif alg==2: # 0.000315 / 0.76723
        patches = np.zeros((height, width, (2*r+1)**2), dtype=np.uint16) # dtype = int becuase it's handled before filterinig  
        img_pad = impad(img, [r,r])
        for y in range(2*r+1):
            for x in range(2*r+1):
                patches[:,:,(2*r+1)*y+x] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width) 
        patches = patches[:height-2*r, :width-2*r, :]
        
        img_conv = np.sum(patches[:,:,:] * kernel.flatten(), axis=2, dtype=dtype) 
    
    elif alg==3:# 0.000161 / 0.010506
        dtype = np.uint16 if np.min(kernel) >= 0 else np.int16 
        img_conv = cv2.filter2D(img.astype(dtype), ddepth=-1, kernel=kernel)
        
    return img_conv 

def imdown(img, factor):
    return img[::int(factor), ::int(factor)]  

def impad(img, pad=[1,1]):
    img_pad = np.zeros((img.shape[0]+2*int(pad[0]), img.shape[1]+2*int(pad[1]))) 
    img_pad[int(pad[0]):-int(pad[0]), int(pad[1]):-int(pad[1])] = img[:] 
    return img_pad

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 

def imshow(img, name='img', size=(30,10)):
    plt.subplots(figsize=size) 
    plt.imshow(img, cmap = 'gray') 
    plt.title(name)
    plt.show() 

def imcompare(img1, img2, name1='img1', name2='img2', size=(20,10)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ax1.imshow(img1, cmap = 'gray') 
    ax1.set_title(name1)
    ax2.imshow(img2, cmap = 'gray') 
    ax2.set_title(name2)
    plt.show() 

def imsave(imgname, img):
    plt.imsave(imgname, img) 

def iminfo(img, name):
    print('*'*5 + name + '*'*5)
    print("type:", type(img))
    print("size:", img.shape) 
    print() 

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    for pt in pts:
        img = cv2.circle(img, (round(pt[0]), round(pt[1])), r, color, thickness)
    return img

# tests 

num = 20 

def run_test_imgrad():
    print('-'*num + 'test Image Gradient' + '-'*num)

    kernel = GradientFilter(method='Sobel') 
    print("sum gradient filter =", np.sum(kernel))
    #imshow(kernel, size=(15,5), name='Sobel filter')

    N = 15*15
    img = np.random.randint(255, size=(N,N))
    #imshow(img) 

    for i in [-1,0,1,2,3]:
        t0 = time.time() 
        g = imgrad(img, kernel, alg=i)
        t1 = time.time() 
        print(f'alg={i}: %f' % (t1 -t0))
        print(g.shape)
        #imshow(g[:,:,0], name='{}'.format(i))
    
    '''
    sum filter = 0
        alg=-1: 0.595809
        (225, 225, 2)
        alg=0: 0.645288
        (225, 225, 2)
        alg=1: 0.628246
        (223, 223, 2)
        alg=2: 0.005698
        (223, 223, 2)
        alg=3: 0.000516
        (225, 225, 2)
    '''


def run_test_gaussian():
    print('-'*num + 'test Gaussian Filter' + '-'*num)

    kernel = GaussianFilter(sigma=1.6, r=1)

    print("sum gaussian filter =", np.sum(kernel))

    #imshow(kernel, size=(15,5), name='Gaussian filter')

    N = 15 * 15
    img = np.random.randint(255, size=(N,N))

    for i in [0,1,2,3]:
        t0 = time.time() 
        g = imconv(img, kernel, alg=i)
        t1 = time.time() 
        print(f'alg={i}: %f' % (t1 -t0))
        print(g.shape)
        #imshow(g, name='{}'.format(i))
    
    

if __name__=='__main__':
    main()
