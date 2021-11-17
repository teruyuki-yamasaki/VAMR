import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import os 

def main():

    args = {
        'img_dir': '../data',
        'lambda': 4,
        'grad_filter': 'Sobel',
        'score': 'Harris',
        'num_kpts': 200,
        'non_max_suppression_radius': 9,
        'descriptor_radius': 8,
        'Mpatch_radius': 1
    }

    filenames = sorted(glob.glob(os.path.join(args['img_dir'], '*')))

    if 0: # show the first two images 
        imcompare(imread(filenames[0]), imread(filenames[1])) 
    
    if 0: # show key points in the fist image 
        for i in range(10):
            img = imread(filenames[i])
            kpts, des = extractKptDes(img, args)
            img = impoints(img, kpts)
            imshow(img) 
    
    if 1:
        img_refer = imread(filenames[0]) 
        for i in range(len(filenames)):
            img_query = imread(filenames[i+1]) 
            showMatches(img_query, img_refer, args) 
            img_refer = img_query

def showMatches(img_query, img_refer, args):
    kpts_query, des_query = extractKptDes(img_query, args) 
    kpts_refer, des_refer = extractKptDes(img_refer, args) 
    matches = matchDescriptors(des_query, des_refer, args['lambda']) 
    #print(matches) 
    img_query = immatches(img_query, matches, kpts_query, kpts_refer)
    imshow(img_query)

def immatches(img, matches, query_kpts, base_kpts):
    qs = query_kpts[matches[0]] 
    ds = base_kpts[matches[1]]
    
    img = impoints(img, qs, color=(0,255,0)) 
    img = impoints(img, ds, color=(0,0,255)) 
    img = imlines(img, zip(swapXY(qs), swapXY(ds)), color=(0,255,0)) 

    return img 

def showMatches0(img_query, img_refer, args):
    kpts_query, des_query = extractKptDes(img_query, args) 
    kpts_refer, des_refer = extractKptDes(img_refer, args) 
    matchMat, matchId = matchDescriptors(des_query, des_refer, args['lambda']) 
    img_query = immatches(img_query, matchId, kpts_query, kpts_refer)
    imshow(img_query)

def immatches0(img, matchId, query_kpts, base_kpts):
    qs = query_kpts[matchId // base_kpts.shape[0]] 
    ds = base_kpts[matchId % base_kpts.shape[0]]
    
    img = impoints(img, qs, color=(0,255,0)) 
    img = impoints(img, ds, color=(0,0,255)) 
    img = imlines(img, zip(swapXY(qs), swapXY(ds)), color=(0,255,0)) 

    return img 

def extractKptDes(img, args):
    '''extracts key points and descriptors from a given image

        - dI(h-2*r, w-2*r, 2) <- img(h, w): r = the radius of image gradient filter 
        - M(h-2*r-2*Mr, w-2*r-2*Mr, (2*r+1)**2) <- dI(h-2*r, h-2*r, 2): Mr = the patch radius of M 
        - R(h-2*r-2*Mr, w-2*r-2*Mr) <- M(h-2*r-2*Mr, w-2*r-2*Mr, (2*r+1)**2): score 

    '''
    
    ### TODO: calculate corner detection scores ###

    # create patches of Ix*Ix, Ix*Iy, Iy*Iy 
    dI = imgrad(img, args['grad_filter'])               # image gradient: Ix, Iy 
    M = StructureTensorM(dI, args['Mpatch_radius'])     # patches of Ix**2, Ix*Iy, Iy**2 

    # compute scores 
    score = args['score'] 
    if score=="Harris":
        R = HarrisScore(M) 
    elif score=="ShiTomasi":
        R = ShiTomasiScore(M) 

    print('img',img.shape) 
    print('dI', dI.shape) 
    print('M', M.shape) 
    print('R', R.shape) 
    
    ### TODO: select key points and obtain descriptors of them ### 
    kpts = selectKeyPoints(R, img.shape, args['num_kpts'], args['non_max_suppression_radius']) 
    dess = describeKeyPoints(img, kpts, args['descriptor_radius'])
    
    return kpts, dess 

def matchDescriptors(query, refer, match_lambda=4):
    '''Given query discriptors and reference discriptors, find discriptor correspondances
    Input:
        - query(Q, (2*r+1)**2): discriptors of key points in query data 
        - refer(D, (2*r+1)**2): discriptors of key points in referance data  
        - match_lambda: the const for threshold 

    Output:
        - descriptor(N, 2*discriptor+1): discriptor 
    '''

    ###TODO: crate a look-at table of matches between discriptors of query and base ###

    #matchMat = np.zeros((query.shape[0], refer.shape[0]), dtype=float)
    matchMat = np.zeros((query.shape[0], refer.shape[0]), dtype=np.uint32) 

    print("Q.shape", query.shape) 

    for i in range(query.shape[0]):
        # compute distance (SSD) between the i-th discriptor in query and each discriptor in base 
        # SSD: the sum of squared differences 
        dist = np.sqrt( np.sum((query[i].reshape(1,-1) - refer)**2, axis=1) ) 

        # regard the i-th discriptor in query as correspondent with a base discriptor
        # if their distance is the smallest 
        dist[dist!=np.min(dist)] = 0 # -> we get e.g., dist = [0. 0. ... 388.462356 0. ... 0.] can be round to integers 
        # dist = dist.astype(np.uint32) # no need to specify datatype here since already done 
        matchMat[i, :] = dist
        # print(dist) 
        # Are we assuming that the exact correspondance does not occur?
    
    ###TODO: set a constraint that reduces unrealistic matches 

    #print(np.max(matchMat)) # 1603: matches such large a min dist are very likely to cause incorrect matches 
    
    min_nonzero_dist = np.min(matchMat[matchMat!=0])            # X[X!=0] extracts nonzero elements of X
    matchMat[matchMat >= match_lambda * min_nonzero_dist] = 0   #  

    #print(min_nonzero_dist) # 115
    #print(np.max(matchMat)) # 456

    # remove double matches
    _, matchId = np.unique(matchMat, return_index=True)

    matchId = matchId[1:] 
    matches = np.zeros((2, matchId.shape[0]), dtype=np.uint32)
    matches[0] = matchId // refer.shape[0]
    matches[1] = matchId % refer.shape[0] 

    '''numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)[source]
        Find the unique elements of an array.

        ararray_like
            Input array. Unless axis is specified, this will be flattened if it is not already 1-D.

        return_indexbool, optional
            If True, also return the indices of ar (along the specified axis, if provided, or in the flattened array) that result in the unique array.

        In this case:
        matchId = [    0 18554 35709 29639 38880 15986 31248  5980 10129 23871 22786  7031
        31780 28096 36164 33260 32935  6296 32149  4520 30365 30500 19053 16761
        39069 29169 35569 11969  4756 20907 10593  4398 38398 27764 28465  6408
        30969 25994 37828 14129 21965 29457 19798  4135 16193 31535  2713 23658
        35939 11421   929  2500  7594 26566 11127  5249  5598 31857  3792 34513
        535  8456 24007 16449 20261 31075  3913 18049  1913 30759 12616 21249
        667  6913 34113 22253 16263 37498 10884  2387 39595  9053 23539  6113
        29894]
    '''

    #return matchMat, matchId[1:]
    return matches 

def describeKeyPoints(img, kpts, descriptor_r=9):
    '''Given an image and its key positions, create descriptors of the key points 
    # 
    # Input:
    #   - img(height,width): image in gray scale 
    #   - kpts(N, 2): pixel coordinates (y,x) of the key points 
    #   - discriptor_r: the patch radius for key point description 
    #
    # Output:
    #   - descriptor(N, (2*discriptor+1)**2): discriptor 
    ''' 

    r = descriptor_r 
    img_pad = impad(img, [r, r]) 

    descriptor = np.zeros((kpts.shape[0], (2*r+1)**2))
    #print(img.shape) 
    for i, kp in enumerate(kpts):
        #print(kp) 
        descriptor[i, :] = img_pad[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1].flatten()

    return descriptor

def selectKeyPoints(scores, img_shape, num_kpts=200, non_max_suppression_r=8):
    # Given corner detection scores, select key points 
    # 
    # Input:
    #   - scores(height,width): corner detection scores 
    #   - num_kpts: the number of key points, 200 for default 
    #   - non_max_suppression_r: the patch radius for non max suppression of pixels surrounding local maxima 
    #
    # Output:
    #   - kpts(num_kpts, 2): pixel coordinates (y,x) of the key point

    r = non_max_suppression_r
    scores_temp = impad(scores, [r,r]) 

    height, width = scores_temp.shape
    R = (img_shape[0]-scores.shape[0])//2

    if 0: 
        print('scores_temp.shape', scores_temp.shape) 
        print('scores.shape', scores.shape) 
        print('img.shape', img_shape) 

    kpts = np.zeros((num_kpts, 2),dtype=np.uint16) 
    for i in range(num_kpts):
        # get the index of the pixel that gives the largest score among the remaining pixels in scores_temp 
        kp = np.argmax(scores_temp.flatten()) # id in scores_temp 

        # convert the id into the pixel coordinate (y,x) in scores, not in scores_temp    
        kp = np.array([kp//width, kp%width]) - r # scores 

        # save the coordinate as the i-th key point's position 
        kpts[i, :] = kp + R 

        # perform non maximum supresssion around the key point 
        scores_temp[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1] = np.zeros((2*r+1,2*r+1)) # non maximum supresssion 
    
    return kpts 

# score 
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

def StructureTensorM(dI, patch_radius=1): # 0.65
    height, width, _ = dI.shape

    Ix = np.zeros((height, width, (2*patch_radius+1)**2), dtype=np.int32)
    Iy = np.zeros((height, width, (2*patch_radius+1)**2), dtype=np.int32) 
    for y in range(2*patch_radius+1):
        for x in range(2*patch_radius+1):
            Ix[:,:,(2*patch_radius+1)*y + x] = np.roll(dI[:,:,0].flatten(), -(width*y+x)).reshape(height, width)
            Iy[:,:,(2*patch_radius+1)*y + x] = np.roll(dI[:,:,1].flatten(), -(width*y+x)).reshape(height, width)
    Ix = Ix[:height - 2*patch_radius, :width - 2*patch_radius, :]
    Iy = Iy[:height - 2*patch_radius, :width - 2*patch_radius, :]

    M = np.zeros((height-2*patch_radius, width-2*patch_radius, 3), np.int32)
    M[:,:,0] = np.sum(Ix**2, axis=-1)
    M[:,:,1] = np.sum(Ix*Iy, axis=-1)
    M[:,:,2] = np.sum(Iy**2, axis=-1) 
    
    return M

def StructureTensorM_for(dI, patch_radius=1): # 11.20 # なぜ結果が変わった？
    height, width, _ = dI.shape

    M = np.zeros((height-2*patch_radius, width-2*patch_radius, 3))  
    for y in range(M.shape[0]):
        for x in range(M.shape[1]): 
            Ix = dI[y:y+2*patch_radius+1, x:x+2*patch_radius+1,0]  
            Iy = dI[y:y+2*patch_radius+1, x:x+2*patch_radius+1,1] 
            M[y,x,:] = [np.sum(Ix**2), np.sum(Ix*Iy), np.sum(Iy**2)]

    return M

def StructureTensorM_cv2(dI, patch_radius=1):  
    Ixx = dI[:,:,0]**2
    Ixy = dI[:,:,0]*dI[:,:,1] 
    Iyy = dI[:,:,1]**2

    patch = np.ones((2*patch_radius+1, 2*patch_radius+1)) 
    M = np.zeros((dI.shape[0], dI.shape[1], 3), dtype=float) 
    M[:,:,0] = cv2.filter2D(Ixx, ddepth=-1, kernel=patch) 
    M[:,:,1] = cv2.filter2D(Ixy, ddepth=-1, kernel=patch) 
    M[:,:,2] = cv2.filter2D(Iyy, ddepth=-1, kernel=patch) 

    return M


# imlags 
def imgrad(img, method='Sobel', alg=1): 
    height, width = img.shape 
    op, opT = imgradop(method)   
    kernel_radius = op.shape[0] // 2
    dI = np.zeros((height-2*kernel_radius, width-2*kernel_radius, 2), dtype=np.int16) 

    #replace this with a faster algorithm 
    if alg==0: # 7.13
        #opT = op.T 
        for y in range(dI.shape[0]):
            for x in range(dI.shape[1]): 
                I = img[y:y+2*kernel_radius+1, x:x+2*kernel_radius+1]  
                dI[y, x, 0] = np.sum(I * op)
                dI[y, x, 1] = np.sum(I * opT)

    elif alg==1: # 0.07
        #opT = op.T 
        patches = np.zeros((height, width, (2*kernel_radius+1)**2), dtype=np.uint16)
        for y in range(2*kernel_radius+1):
            for x in range(2*kernel_radius+1):
                patches[:,:,(2*kernel_radius+1)*y+x] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width)
        patches = patches[:height-2*kernel_radius, :width-2*kernel_radius, :]

        dI[:,:,0] = np.sum(patches[:,:,:] * op.flatten(), axis=-1)
        dI[:,:,1] = np.sum(patches[:,:,:] * opT.flatten(), axis=-1)

    return dI

def imgrad_cv2(img, method='Sobel'):
    opX, opY = imgradop(method)  

    dI = np.zeros((len(img), len(img[0]), 2), dtype=float) 
    dI[:,:,0] = cv2.filter2D(img, ddepth=-1, kernel=opX)
    dI[:,:,1] = cv2.filter2D(img, ddepth=-1, kernel=opY)
    
    return dI 

def imgradop(method='Sobel'):
    if method=="Sobel":
        op = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
        
    elif method=="Prewitt":
        op = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    
    return op, op.T 

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 

def imresize(img, factor=2):
    return img[::factor, ::factor]

def imshow(img, name="img"):
    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break
    cv2.destroyAllWindows()

def impad(img, pad=(1,1)):
    img_pad = np.zeros((len(img)+2*pad[0], len(img[0])+2*pad[1]), dtype=float)
    img_pad[pad[0]:-pad[0], pad[1]:-pad[1]] = img 
    return img_pad 

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    for pt in pts: img = cv2.circle(img, (round(pt[1]), round(pt[0])), r, color, thickness)
    return img

def imlines(img, lines, color=(0,0,255), thickness=2):
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    for line in lines: img = cv2.line(img, tuple(line[0]), tuple(line[1]), color=color, thickness=thickness) 
    return img 

def imcompare(img1, img2):
    img = np.vstack((img1, img2))
    imshow(img) 

def imsave(imgname, img):
    plt.imsave(imgname, img) 

def iminfo(img, name):
    print('*'*5 + name + '*'*5)
    print("type:", type(img))
    print("size:", img.shape) 
    print() 

def swapXY(A):
    B = np.zeros_like(A) 
    B[:,0] = A[:,1]
    B[:,1] = A[:,0] 
    return B


if __name__=='__main__':
    main() 
