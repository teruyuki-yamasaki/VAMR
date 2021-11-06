import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def imread(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 

def imshow(img, name='img'):
    cv2.imshow(name, img) 

    while True:
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows() 

def imshowplt(img, name='img', size=(60,20)):
    plt.subplots(figsize=size) 
    plt.imshow(img) 
    plt.title(name)
    plt.show() 

def imgradOp(method='Sobel'):
    if method=='Sobel':
        op = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    elif method=='Prewitt':
        op = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) 
    
    return op, op.T

def imgradXY(img, method='Sobel'):
    H, W = img.shape 
    opX, opY = imgradOp(method) 
    kernel_size = len(opX)

    dI = np.zeros((H-kernel_size+1, W-kernel_size+1, 2))  

    for v in range(len(dI)):
        for u in range(len(dI[0])): 
            I = img[v:v+kernel_size, u:u+kernel_size]  
            dI[v, u, 0] = np.sum(I * opX)
            dI[v, u, 1] = np.sum(I * opY)
    
    return dI

def StructureTensorM(dI, patch_size):
    M = np.zeros((len(dI)-patch_size+1, len(dI[0])-patch_size+1, 3))  
    for v in range(len(M)):
        for u in range(len(M[0])): 
            Ix = dI[v:v+patch_size,u:u+patch_size,0] 
            Iy = dI[v:v+patch_size,u:u+patch_size,1] 
            M[v,u] = [np.sum(Ix*Ix), np.sum(Ix*Iy), np.sum(Iy*Iy)]
    return M

def ShiTomasiScore(M, H, W):
    R = np.zeros((H, W))
    h = int((H - len(M))/2)
    w = int((W - len(M[0]))/2)

    e1 = M[:, :, 0]
    e2 = M[:, :, 1]
    e3 = M[:, :, 2] 

    R = 0.5 * (e1 + e3 - np.sqrt((e1 - e3)**2 + 4*e2**2))    

    return R

def HarrisScore(M, H, W, kappa=0.08):
    e1 = M[:, :, 0]
    e2 = M[:, :, 1]
    e3 = M[:, :, 2] 

    detM = e1*e3 - e2**2
    trM = e1 + e3 

    R = detM - kappa * trM**2

    return R 

def histgram(X, a=0, b=1, N=1000, ymax=1e4):
    counts, bins = np.histogram(X.flatten(),bins=N)
    plt.hist(bins[:-1], bins[int(a*N):int(b*N)], weights=counts)
    plt.text(10,10,'Rmax={}'.format(np.max(X))) 
    plt.ylim(-0.1,ymax)
    plt.show() 

def preproess(R, K):
    R = R * (R > K)
    Rmax = np.max(R) 
    Rmin = np.min(R)
    print(Rmax, Rmin) 
    return (R - Rmin) / (Rmax - Rmin) * 255.0

def NonMaxSupression(X, r=2):
    for v in range(len(X)-2*r):
        for u in range(len(X[0])-2*r):
            Xsub = X[v:v+2*r+1, u:u+2*r+1]
            Xmax = np.max(Xsub)
            X[v:v+2*r+1,u:u+2*r+1] = Xmax * (Xsub == Xmax) 
    return X 

def selectKeyPoints(X, r=4, k=50):
    X = NonMaxSupression(X, r)
    bound = sorted(X.flatten(), reverse=True)[k]
    P = np.where(X >= bound) 
    return np.concatenate([P[0], P[1]], axis=0).reshape(2,-1).T

def selectKeyPointsOrder(X, r=4, k=50):
    X = NonMaxSupression(X, r)
    bound = sorted(X.flatten(), reverse=True)[:k]

    pts = np.zeros((k, 2), dtype=np.uint16) 
    i = 0 
    for b in bound:
        P = np.where(X == b) 
        P = np.concatenate([P[0], P[1]], axis=0).reshape(2,-1).T
        j = i + len(P)
        pts[i:j] = P 
        i = j 
    
    return pts 

def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

def add_points(img, pts, r=2, color=(0,0,255), thickness=-1):
    if len(img.shape) == 2:
        img = gray2rgb(img) 

    for pt in pts:
        img = cv2.circle(img, (round(pt[1]), round(pt[0])), r, color, thickness)
    return img

def padding(img, r):
    if len(img.shape) == 3:
        img = rgb2gray(img)
    Img = np.zeros((len(img)+2*r, len(img[0])+2*r)) 
    Img[r:-r, r:-r] = img
    return Img

def descriptor(img, kpt, r):
    return img[kpt[0]:kpt[0]+2*r+1, kpt[1]:kpt[1]+2*r+1].flatten()

def describeKeyPoints(img, keypoints, r):
    img = padding(img, r)
    X = np.zeros((len(keypoints), (2*r+1)**2))
    for i, kpt in enumerate(keypoints):
        X[i] = descriptor(img, kpt, r)
    return X

def recover(x):
    return x.reshape( int( np.sqrt(len(x)) ),-1)

def img2KeyPoints(img):
    H, W = img.shape
    dI = imgradXY(img)
    M = StructureTensorM(dI, patch_size=5)
    R = HarrisScore(M, H, W) 
    R = preproess(R, K=5e11)
    P = selectKeyPointsOrder(R, k=200)
    return P

def matchDescriptors(query, database, const):
    ans = np.zeros(len(query), dtype=int)
    dmin = 1e5
    for i in range(len(query)):
        diff = query[i] - database 
        ssd = np.sum(diff**2, axis=1) # sum of squared differnece 
        dmin = min(np.min(ssd), dmin)  

    for i in range(len(query)):
        diff = query[i] - database 
        ssd = np.sum(diff**2, axis=1) # sum of squared differnece 
        v = np.argmin(ssd) 
        ans[i] = v if v not in ans and np.min(ssd) < const*dmin else -1
    return ans 

def final(img0, img1, P0, P1, ans, r=5, const=1.5):
    img1 = add_points(img1, P0, color=(0, 0, 255))
    img1 = add_points(img1, P1, color=(0, 255, 0))

    d0 = describeKeyPoints(img0, P0, r)
    d1 = describeKeyPoints(img1, P1, r) 
    ans = matchDescriptors(d1, d0, const)

    print(ans)

    lines = np.zeros((np.count_nonzero(ans!=-1), 2,2), dtype=int)
    #print(np.count_nonzero(ans!=-1))
    id = 0
    #print(len(P1))
    for i in range(len(ans)):
        if ans[i] != -1:
            lines[id][0] = P1[i]
            #print(i, ans[i])
            lines[id][1] = P0[ans[i]]
            id += 1
    
    for i in range(len(lines)):
        xy0 = (lines[i,0,1], lines[i,0,0])
        xy1 = (lines[i,1,1], lines[i,1,0])

        img1 = cv2.line(img1, xy0, xy1, (0,255,0), 2)
    
    imshow(img1)


def main():
    img = imread('../data/000000.png')
    imshow(img)

    dI = imgradXY(img)
    imshow(dI[:,:,0], 'Ix') 
    imshow(dI[:,:,1], 'Iy') 

    M = StructureTensorM(dI, patch_size=5)
    imshow(M[:,:,0]/np.linalg.norm(M[:,:,0])*255, 'Ix**2')
    imshow(M[:,:,1]/np.linalg.norm(M[:,:,1])*255, 'Ix*Iy')
    imshow(M[:,:,2]/np.linalg.norm(M[:,:,2])*255, 'Iy**2')

    H, W = img.shape
    R1 = ShiTomasiScore(M, H, W) 
    histgram(R1,a=0.001,b=0.2)
    R1 = preproess(R1, K=5e5)
    histgram(R1,a=0.001,ymax=256) 
    imshow(R1, 'Shi-Tomashi')  

    R2 = HarrisScore(M, H, W) 
    histgram(R2, a=0.25,b=0.75)
    R2 = preproess(R2, K=5e11)
    histgram(R2, a=0.001, ymax=256) 
    imshow(R2, 'Harris') 

    P = selectKeyPointsOrder(R1, k=200)
    print(img.shape)

    img_keys = add_points(img, P)

    imshow(img_keys) 

    imshow(padding(img,r=4))

    X = describeKeyPoints(img, P, r=16)

    #for x in X[:10]: imshow(recover(x), size=(4,4))

    img0 = imread('../data/000000.png')
    img1 = imread('../data/000001.png') 

    imshow(img0)
    imshow(img1)

    P0 = img2KeyPoints(img0)
    P1 = img2KeyPoints(img1)

    d0 = describeKeyPoints(img0, P0, r=5)
    d1 = describeKeyPoints(img1, P1, r=5) 

    #imshow(recover(d1[0]), size=(5,5))

    diff = d1[0] - d0
    print(np.sum(diff**2, axis=1).shape)

    ans = matchDescriptors(d1, d0, 50)
    print(ans)

    final(img0, img1, P0, P1, ans, const=3.5)


if __name__=="__main__":
    main() 
