import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def imread(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 

def imshow(img, name="img"):
    cv2.imshow(name, img) 

    while True:
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows() 

def add_points(img, pts, r=5, color=(0,0,255), thickness=-1):
    for pt in pts:
        img = cv2.circle(img, (round(pt[0]), round(pt[1])), r, color, thickness)
    return img 

def ImageGradientOp(method='Sobel'):
    if method=='Sobel':
        X = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])
        Y = X.T 

    elif method=='Prewitt':
        X = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
        Y = X.T 
    
    return X, Y

def ImageGradientXY(img, method='Sobel'):
    H, W = img.shape 
    opX, opY = ImageGradientOp(method) 
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

    for v in range(len(M)):
        for u in range(len(M[0])):
            e1, e2, e3 = M[v, u]
            R[v+h,u+w] = 0.5 * (e1 + e3 - np.sqrt((e1 - e3)**2 + 4*e2**2))    

    return R

def HarrisScore(M, H, W, kappa=0.08):
    R = np.zeros((H, W))
    h = int((H - len(M))/2)
    w = int((W - len(M[0]))/2)

    for v in range(len(M)):
        for u in range(len(M[0])):
            e1 = M[v, u, 0]
            e2 = M[v, u, 1]
            e3 = M[v, u, 2] 
            detM = e1*e3 - e2**2
            trM = e1 + e3 
            R[v+h,u+w] = detM - kappa * trM**2

    return R 

def preproess(R, K):
    R = R * (R > K)
    Rmax = np.max(R) 
    Rmin = np.min(R)
    print(Rmax, Rmin) 
    return (R - Rmin) / (Rmax - Rmin) * 255.0


def histgram(X, a=0, b=1, N=1000, ymax=1e4):
    counts, bins = np.histogram(X.flatten(),bins=N)
    plt.hist(bins[:-1], bins[int(a*N):int(b*N)], weights=counts)
    plt.text(10,10,'Rmax={}'.format(np.max(X))) 
    plt.ylim(-0.1,ymax)
    plt.show() 

def keypoints(R, k=50):
    R_sort = sorted(R.flatten(),reverse=True)
    print(k) 
    K = R_sort[k]
    return np.where(R >= K)

patch_size = 5
def main():
    img = imread('../data/000000.png')
    imshow(img, '000000.png')
    H, W = img.shape; print(img.shape); 

    dI = ImageGradientXY(img) 
    imshow(dI[:,:,0], 'Ix') 
    imshow(dI[:,:,1], 'Iy') 
    h, w, _ = dI.shape; print(dI.shape) 

    M = StructureTensorM(dI, patch_size) 
    imshow(M[:,:,0]/np.linalg.norm(M[:,:,0])*255, 'Ix**2')
    imshow(M[:,:,1]/np.linalg.norm(M[:,:,1])*255, 'Ix*Iy')
    imshow(M[:,:,2]/np.linalg.norm(M[:,:,2])*255, 'Iy**2')

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

    K1 = keypoints(R1) 
    img1 = add_points(img, K1)
    imshow(img1, "ShiTomashi")
    
    K2 = keypoints(R2) 
    img2 = add_points(img, K2)
    imshow(img2, "Harris")
    

if __name__=="__main__":
    main() 
