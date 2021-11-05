import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 

def imresize(img, factor=2):
    return img[::factor, ::factor]

def imshow(img, name="img"):
    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break 
    cv2.destroyAllWindows()

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    for pt in pts: img = cv2.circle(img, (round(pt[1]), round(pt[0])), r, color, thickness)
    return img

def imlines(img, lines, color=(0,0,255), thickness=2):
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    for line in lines: img = cv2.line(img, tuple(line[0]), tuple(line[1]), color=color, thickness=thickness) 
    return img 

def imcompare(img1, img2, size=(30,20)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ax1.imshow(img1, cmap = 'gray') 
    ax1.set_title('img_1')
    ax2.imshow(img2, cmap = 'gray') 
    ax2.set_title('img_2')
    plt.show() 

def imsave(imgname, img):
    plt.imsave(imgname, img) 

def iminfo(img, name):
    print('*'*5 + name + '*'*5)
    print("type:", type(img))
    print("size:", img.shape) 
    print() 

def impad(img, pad=(1,1)):
    img_pad = np.zeros((len(img)+2*pad[0], len(img[0])+2*pad[1]), dtype=float)
    img_pad[pad[0]:-pad[0], pad[1]:-pad[1]] = img 
    return img_pad 

def imgradOp(method='Sobel'):
    if method=="Sobel":
        op = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
        
    elif method=="Prewitt":
        op = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    
    return op, op.T 

def imgrad(img, method='Sobel'):
    opX, opY = imgradOp(method)  

    dI = np.zeros((len(img), len(img[0]), 2), dtype=float) 
    dI[:,:,0] = cv2.filter2D(img, ddepth=-1, kernel=opX)
    dI[:,:,1] = cv2.filter2D(img, ddepth=-1, kernel=opY)
    
    return dI 

def StructureTensorM(dI, patch_r=1):  
    Ixx = dI[:,:,0]**2
    Ixy = dI[:,:,0]*dI[:,:,1] 
    Iyy = dI[:,:,1]**2

    patch = np.ones((2*patch_r+1, 2*patch_r+1)) 
    M = np.zeros((len(dI), len(dI[0]), 3), dtype=float) 
    M[:,:,0] = cv2.filter2D(Ixx, ddepth=-1, kernel=patch) 
    M[:,:,1] = cv2.filter2D(Ixy, ddepth=-1, kernel=patch) 
    M[:,:,2] = cv2.filter2D(Iyy, ddepth=-1, kernel=patch) 
    return M

def ShiTomasiScore(M):
    e1 = M[:, :, 0]
    e2 = M[:, :, 1]
    e3 = M[:, :, 2] 

    # score = min(lambda_1, lambda_2) where lambdas are eigenvalues of M = [[e1 e2] [e2 e3]]
    scores = 0.5 * (e1 + e3 - np.sqrt((e1 - e3)**2 + 4*e2**2)) 
    scores[scores<0] = 0

    return scores 

def HarrisScore(M, kappa=0.08):
    e1 = M[:, :, 0]
    e2 = M[:, :, 1]
    e3 = M[:, :, 2] 

    detM = e1*e3 - e2**2
    trM = e1 + e3 

    scores = detM - kappa * trM**2
    scores[scores<0] = 0

    return scores 

def selectKeyPoints(scores, num_kpts=50, non_max_suppression_r=1):
    r = non_max_suppression_r
    scores_temp = impad(scores, [r,r]) 
    W = len(scores_temp[0])

    kpts = np.zeros((num_kpts, 2),dtype=int) 
    for i in range(num_kpts):
        kp = np.argmax(scores_temp.flatten()) # index 
        kp = np.array([kp//W, kp%W]) # index to loc 
        kpts[i, :] = kp - r
        scores_temp[kp[0]-r:kp[0]+r+1, kp[1]-r:kp[1]+r+1] = np.zeros((2*r+1,2*r+1)) # non maximum supresssion 
    
    return kpts 

def describeKeyPoints(img, kpts, descriptor_r=8):
    r = descriptor_r 
    descriptor = np.zeros((len(kpts), (2*r+1)**2))
    img_pad = impad(img, [r, r]) 
    for i, kp in enumerate(kpts):
        descriptor[i, :] = img_pad[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1].flatten()
    return descriptor

def matchDescriptors(query, base, match_lambda=4):
    matchMat = np.zeros((len(query), len(base))) 
    for i, q in enumerate(query):
        dist = np.sqrt(np.sum((q - base)**2, axis=1)) 
        matchMat[i, :] = dist * (dist==np.min(dist)) 
    
    min_nonzero_dist = np.min(matchMat[matchMat > 0])
    matchMat[matchMat >= min_nonzero_dist*match_lambda] = 0

    _, matchId = np.unique(matchMat, return_index=True)

    return matchMat, matchId

def swapXY(A):
    B = np.zeros_like(A) 
    B[:,0] = A[:,1]
    B[:,1] = A[:,0] 
    return B

def immatches(img, matchId, query_kpts, base_kpts):
    qs = query_kpts[matchId // len(base_kpts)]
    ds = base_kpts[matchId % len(base_kpts)]
    
    img = impoints(img, qs, color=(0,255,0)) 
    img = impoints(img, ds, color=(0,0,255)) 
    img = imlines(img, zip(swapXY(qs), swapXY(ds)), color=(0,255,0)) 

    return img 

def img2descriptors(img, num_kpts=200, non_max_suppression_r=1, descriptor_r=8, score="Harris", gradfilter="Sobel"):
    dI = imgrad(img, gradfilter) 
    M = StructureTensorM(dI, patch_r=1) 

    if score=="Harris":
        R = HarrisScore(M) 
    elif score=="ShiTomasi":
        R = ShiTomasiScore(M) 
    
    kpts = selectKeyPoints(R, num_kpts, non_max_suppression_r) 
    dess = describeKeyPoints(img, kpts, descriptor_r)
    
    return kpts, dess 

def getFileNames():
    return sorted(glob.glob('../data/*'))

def main():
    img0 = imread('../data/000000.png')
    img1 = imread('../data/000001.png') 

    imshow(img0, 'img0')
    imshow(img1, 'img1') 

    kpts0, des0 = img2descriptors(img0) 
    kpts1, des1 = img2descriptors(img1)

    matchMat, matchId = matchDescriptors(des1, des0, match_lambda=4)  
    imshow(matchMat)
    imshow(immatches(img1, matchId, kpts1, kpts0))

    filenames = getFileNames() 
    img_base = imread(filenames[0]) 
    for i in range(len(filenames)):
        img_query = imread(filenames[i+1]) 

        kpts_query, des_query = img2descriptors(img_query) 
        kpts_base, des_base = img2descriptors(img_base) 
        matchMat, matchId = matchDescriptors(des_query, des_base, match_lambda=4) 
        imshow(immatches(img_query, matchId, kpts_query, kpts_base))

        img_base = img_query

if __name__=='__main__':
    main() 
