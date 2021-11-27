import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import gaussian_filter,convolve
from scipy.spatial.distance import cdist 
import glob 
import os 
import time 

def main():
    args = {
        'img_names': '../data/*.png',
        'lambda': 4,
        'kappa': 0.08, 
        'grad_filter': 'Sobel',
        'score': 'Harris',
        'num_kpts': 200,
        'non_max_suppression_radius': 9,
        'descriptor_radius': 8,
        'corner_patch_radius': 4
    }

    filenames = sorted(glob.glob(args['img_names'])) 

    if 1:
        run_test_imgrad(filenames[0])
    
    if 1:
        run_test_imstructure(filenames[0])

    if 0:
        run_test_scores(filenames[0])
    
    if 0:
        run_test_selectKeyPoints('../data/images/CircleLineRect.png')
    
    if 0:
        run_test_selectKeyPoints(filenames[0])
    
    if 0:
        run_test_describeKeyPoints(filenames[0])
    
    if 0:
        run_test_matchDescriptors(['../data/images/CircleLineRect.png'])
    
    if 1:
        run_test_matchDescriptors(filenames, num_kpts=300, mode=1)
        run_test_matchDescriptors(filenames, num_kpts=200, mode=0)
    
    if 1:
        run_all(filenames)  

    if 0:
        run_test_cdist()

def run_test_matchDescriptors(filenames, num_kpts=10, mode=0): 
    #---------------------------------------------------------
    # set parameters 
    #---------------------------------------------------------

    # flags 
    gradalg = 1
    kptalg = 0 
    matchalg = 2 

    # filenames 
    filename = filenames[0]
    name = filename.split('/')[-1].split('.')[0]

    #---------------------------------------------------------
    # load images 
    #---------------------------------------------------------
    
    img0 = imread(filename)

    if mode==0:
        img1 = np.roll(img0, img0.shape[1]*20+30) 
    elif mode==1:
        img1 = imread(filenames[1])  

    imshow(img0) 
    imshow(img1) 

    #---------------------------------------------------------
    # obtain key points and descriptors 
    #---------------------------------------------------------

    kpts0, des0 = im2kptdes(img0, num_kpts, gradalg, kptalg) 
    kpts1, des1 = im2kptdes(img1, num_kpts, gradalg, kptalg)  

    # show kpts 
    if 0:
        print('\n kpts0 = \n', kpts0) 
        print('\n kpts1 = \n', kpts1)
    
    # img0 
    imshow(impoints(img0, kpts0))

    # img1 
    img1 = impoints(img1, kpts1, color=(0,255,0))
    img1 = impoints(img1, kpts0, color=(0,0,255))
    imshow(img1)

    # show descriptors 
    imdescriptors(des0, 'des0')
    imdescriptors(des1, 'des1')  

    #---------------------------------------------------------
    # match descriptors 
    #---------------------------------------------------------
    # find matches 
    matches = matchDescriptors(des1, des0, match_lambda=4, alg=matchalg)
    print(f'\n matches{matches.shape} = \n', matches) 

    # show matches 
    
    if 1:
        img1 = immatches(img1, kpts1, kpts0, matches) 

    else:
        q = kpts1[matches[0]]
        b = kpts0[matches[1]] 
        img1 = imlines(img1, zip(swapXY(q), swapXY(b)), color=(0,255,0))
    imshow(img1, f'matches_{name}_grad{gradalg}_match{matchalg}_mode{mode}', save=True)  


def imdescriptors(descriptors, name):
    #---------------------------------------------------------
    # set parameters 
    #---------------------------------------------------------
    num_descriptors = descriptors.shape[0] 
    d_size = round(descriptors.shape[1] ** 0.5)
    W = num_descriptors//2
    r = 2
    print('d_size = ', d_size) 

    #---------------------------------------------------------
    # visualize descriptors 
    #---------------------------------------------------------
    im = np.zeros((2 *(d_size+2*r), W * (d_size+2*r)), dtype=np.uint8) 
    for i in range(num_descriptors):
        im[(i//W) * (d_size+2*r):(i//W+1) * (d_size+2*r),\
            (i % W) * (d_size+2*r): (i % W+1) * (d_size+2*r)] = impad(descriptors[i].reshape(d_size, d_size), [r,r])
    
    plt_imshow(im, name) 

def im2kptdes(img, num_kpts=10, gradalg=0, kptalg=0):
    '''Obtain the key points and descriptors of a given image 
    '''

    dI = imgrad(img, alg=gradalg).astype(float) 
    M = imstructure(dI, patch_radius=1)
    R = Harris(M) 
    kpts = selectKeyPoints(R, img.shape, num_kpts, non_max_suppression_radius=5, alg=kptalg)
    des = describeKeyPoints(img, kpts, descriptor_radius=4) 
    return kpts, des   


def matchDescriptors(query, refer, match_lambda=4, alg=2):
    '''Given query discriptors and reference discriptors, find discriptor correspondances
    Input:
        - query(Q, (2*r+1)**2): discriptors of key points in query data 
        - refer(D, (2*r+1)**2): discriptors of key points in referance data  
        - match_lambda: the const for threshold 

    Output:
        - matches(2, numMatches): kpt correspondences between query descriptors and reference descriptors  
    '''

    #---------------------------------------------------------------------------------------
    # alg 0: a naive implementation of desriptor matching
    #       - not consider threshold, duplication, etc. 
    #       - implemented using for loops 
    #---------------------------------------------------------------------------------------
    if alg==0:
        ###TODO: crate a look-at table of matches between discriptors of query and base ###
        ssd = np.zeros((query.shape[0], refer.shape[0]), dtype=np.uint32) 
        for i in range(query.shape[0]):
            # compute distance (SSD) between the i-th discriptor in query and each discriptor in base 
            # SSD: the sum of squared differences 
            row = np.sqrt( np.sum( np.power(query[i].reshape(1,-1) - refer, 2.0), axis=1) )        
            ssd[i, :] = row
            #print(ssd)
        
        #print(matchMat) 
            
        ###TODO: set a constraint that reduces unrealistic matches 
        matchargs = np.argmin(ssd, axis=1) 

        num_matches = matchargs.shape[0] 
        matches = np.zeros((2, num_matches), dtype=np.uint16)
        for i in range(num_matches): 
            matches[:,i] = [i, matchargs[i]] 
    
    #---------------------------------------------------------------------------------------
    # alg 1: a naive implementation of desriptor matching
    #       - cdist (scipy) is used to compute ssd 
    #---------------------------------------------------------------------------------------
    elif alg==1:
        ssd = cdist(query, refer, metric='euclidean').astype(np.uint32) 
        matchargs = np.argmin(ssd, axis=1) 

        num_matches = matchargs.shape[0]
        matches = np.zeros((2, num_matches), dtype=np.uint16)
        for i in range(num_matches): 
            matches[:,i] = [i, matchargs[i]] 
        #print(ssd) 

        if 1:
            print(f'\n ssd(min={np.min(ssd)}) = \n', ssd) 
            print(f'\n ssd_min = \n', np.min(ssd, axis=-1, keepdims=True)) 
            print(f'\n match_args = \n', matchargs) 
            
    #---------------------------------------------------------------------------------------
    # alg 2: an adjusted implementation 
    #       - cdist (scipy) is used to compute ssd 
    #       - a thresold is applied
    #       - each descriptor has no more than one match 
    #---------------------------------------------------------------------------------------
    elif alg==2:
        ssd = cdist(query, refer, metric='euclidean') #.astype(np.uint32)        
        mask = (np.min(ssd, axis=-1, keepdims=True) == ssd) # find a match for each query descriptor 
        mask *= (np.min(ssd, axis=0, keepdims=True) == ssd) # do not allow multiple match for each reference descriptor
        mask *= ((np.min(ssd)+1)*match_lambda >= ssd)       # apply a threshold, assumig that min_ssd can be excatly equal to 0: +1 is for that reason. 
        matches = np.vstack(np.where(mask))                 # obtain (y,x) from a boolean matrix 

    return matches 

def describeKeyPoints(img, kpts, descriptor_radius=9):
    '''Given an image and its key positions, create descriptors of the key points 
    # 
    # Input:
    #   - img(height,width): image in gray scale 
    #   - kpts(N, 2): pixel coordinates (y,x) of the key points 
    #   - discriptor_radius: the patch radius for key point description 
    #
    # Output:
    #   - descriptor(N, (2*discriptor+1)**2): discriptor 
    ''' 

    r = descriptor_radius 
    img_pad = impad(img, [r, r]) 

    #---------------------------------------------------------------------------------------
    # reconstruct a small image patch from each descriptor, which is given as a 1D vector 
    #---------------------------------------------------------------------------------------
    descriptor = np.zeros((kpts.shape[0], (2*r+1)**2), dtype=np.uint8) 
    for i, kp in enumerate(kpts):
        descriptor[i, :] = img_pad[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1].flatten()

    return descriptor

def selectKeyPoints(scores, img_shape, num_kpts=200, non_max_suppression_radius=8, alg=0):
    '''Given corner detection scores, select key points 
    # 
    # Input:
    #   - scores(height,width): corner detection scores 
    #   - num_kpts: the number of key points, 200 for default 
    #   - non_max_suppression_r: the patch radius for non max suppression of pixels surrounding local maxima 
    #
    # Output:
    #   - kpts(num_kpts, 2): pixel coordinates (y,x) of the key point
    '''

    #---------------------------------------------------------------------------------------
    # set parameters 
    #---------------------------------------------------------------------------------------
    debugmode = alg 

    r = non_max_suppression_radius
    scores_temp = impad(scores, [r,r]) 
    height, width = scores_temp.shape

    R = (img_shape[0]-scores.shape[0])//2

    if debugmode: 
        print('img.shape = ', img_shape)
        print('scores.shape = ', scores.shape) 
        print('scores_temp.shape = ', scores_temp.shape) 
        print('R = ', R)
        print('nonmaxr = ', r)
        print(f'\n socres_temp{scores_temp.shape} = \n', scores_temp) 
    
    #---------------------------------------------------------------------------------------
    # select key points, while simultaneously applying non-maximum suppression 
    #---------------------------------------------------------------------------------------
    kpts = np.zeros((num_kpts, 2),dtype=np.uint32) 
    for i in range(num_kpts):
        if debugmode: max_temp = np.max(scores_temp) 

        #---------------------------------------------------------------------------------------
        # select as a key point a pixel with the highest score value among the unsuppressed 
        # and obtain its coordinate 
        #---------------------------------------------------------------------------------------
        # get the index of the pixel that gives the largest score among the remaining pixels in scores_temp 
        kp_id = np.argmax(scores_temp.flatten()) # id in scores_temp 

        # convert the id into the pixel coordinate (y,x) in scores_temp    
        kp = np.array([kp_id//width, kp_id%width]) 

        # save the coordinate as the i-th key point's position 
        kpts[i, :] = kp - r + R # r: non_max_sup_r, R: (img.shape[0]-scores.shape[0])//2

        #---------------------------------------------------------------------------------------
        # perform non maximum suppression 
        #--------------------------------------------------------------------------------------- 
        #scores_temp[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1] = np.zeros((2*r+1,2*r+1)) # non maximum supresssion 
        scores_temp[kp[0]-r:kp[0]+r+1, kp[1]-r:kp[1]+r+1] = np.zeros((2*r+1,2*r+1)) # non maximum supresssion 

        if debugmode:
            print(f'\n socres_temp (max == {max_temp}, pos=={kpts[i]}) = \n', scores_temp) 
            print()
        
        if np.max(scores_temp)==0:
            if debugmode: print('we have no more pixels to be selected as key points')
            break 

    return kpts 

#---------------------------------------------------------------------------------------
# Score Calculations 
#---------------------------------------------------------------------------------------

def Harris(M, kappa=0.08, alg=0):

    det = np.multiply(M[0], M[2]) - np.multiply(M[1], M[1])
    trc = np.add(M[0], M[2]) 

    if alg==0:
        H = np.subtract(det, kappa * np.multiply(trc, trc))     
    
    elif alg==1:
        H = det - kappa * np.multiply(trc, trc) 

    elif alg==2:
        H = det - kappa * trc * trc 

    return H 

def ShiTomsashi(M, alg=0):

    if alg==0:
        S = 0.5 * (
            M[0] + M[2] - np.sqrt(\
            np.multiply(M[0]-M[2],M[0]-M[2]) + 4 * np.multiply(M[1], M[1]))
            )
    
    elif alg==1:
        diff = np.subtract(M[0],M[2])
        S = 0.5 * np.subtract(
            np.add(M[0], M[2]), np.sqrt(np.add(
            np.multiply(diff, diff), 4 * np.multiply(M[1], M[1])))
        )
        
    elif alg==2:
        S = 0.5 * (M[0] + M[2] - np.sqrt((M[0] - M[2])**2 + 4*M[1]**2))
    
    elif alg==3:
        S = 0.5 * (M[0] + M[2] - ((M[0] - M[2])**2 + 4*M[1]**2)**0.5)

    return S 



#---------------------------------------------------------------------------------------------------------------
#  imalgs 
#---------------------------------------------------------------------------------------------------------------

def imstructure(dI, patch_radius=1, alg=0):

    # Q: do i need to take a mean of each patch?

    '''
    alg = 0 took 0.0159 sec

    alg = 1 took 0.0148 sec

    alg = 2 took 0.0520 sec

    alg = 3 took 0.0181 sec

    alg = 4 took 0.0177 sec

    '''

    #---------------------------------------------------------------------------------------
    # compute with cv2.filter2D, np.multiply 
    #--------------------------------------------------------------------------------------- 

    if alg==0:
        patch = np.ones((2*patch_radius+1, 2*patch_radius+1)) 
        M = np.zeros((3, dI.shape[1], dI.shape[2])) # dtype ?
        M[0] = cv2.filter2D(np.multiply(dI[0], dI[0]), ddepth=-1, kernel=patch)
        M[1] = cv2.filter2D(np.multiply(dI[0], dI[1]), ddepth=-1, kernel=patch) 
        M[2] = cv2.filter2D(np.multiply(dI[1], dI[1]), ddepth=-1, kernel=patch)
    
    #---------------------------------------------------------------------------------------
    # compute with cv2.filter2D, *
    #--------------------------------------------------------------------------------------- 
    
    elif alg==1:
        patch = np.ones((2*patch_radius+1, 2*patch_radius+1)) 
        M = np.zeros((3, dI.shape[1], dI.shape[2])) # dtype ?
        M[0] = cv2.filter2D(dI[0]*dI[0], ddepth=-1, kernel=patch)
        M[1] = cv2.filter2D(dI[0]*dI[1], ddepth=-1, kernel=patch) 
        M[2] = cv2.filter2D(dI[1]*dI[1], ddepth=-1, kernel=patch)
    
    #---------------------------------------------------------------------------------------
    # compute with np.roll, *  
    #--------------------------------------------------------------------------------------- 
    elif alg==2:
        _, height, width= dI.shape

        Ix = np.zeros(((2*patch_radius+1)**2, height, width), dtype=np.int32)
        Iy = np.zeros(((2*patch_radius+1)**2, height, width), dtype=np.int32) 
        for y in range(2*patch_radius+1):
            for x in range(2*patch_radius+1):
                Ix[(2*patch_radius+1)*y + x] = np.roll(dI[0].flatten(), -(width*y+x)).reshape(height, width)
                Iy[(2*patch_radius+1)*y + x] = np.roll(dI[1].flatten(), -(width*y+x)).reshape(height, width)
        Ix = Ix[:, :height - 2*patch_radius, :width - 2*patch_radius]
        Iy = Iy[:, :height - 2*patch_radius, :width - 2*patch_radius]

        M = np.zeros((3, height-2*patch_radius, width-2*patch_radius), np.int32)
        M[0] = np.sum(Ix*Ix, axis=0) 
        M[1] = np.sum(Ix*Iy, axis=0) 
        M[2] = np.sum(Iy*Iy, axis=0) 
    
    elif alg==3:
        patch = np.ones((2*patch_radius+1, 2*patch_radius+1)) 
        M = np.zeros((3, dI.shape[1], dI.shape[2])) # dtype ?
        M[0] = convolve(dI[0]*dI[0], patch)
        M[1] = convolve(dI[0]*dI[1], patch) 
        M[2] = convolve(dI[1]*dI[1], patch)
    
    elif alg==4:
        patch = np.ones((2*patch_radius+1, 2*patch_radius+1)) 
        M = np.zeros((3, dI.shape[1], dI.shape[2])) # dtype ?
        M[0] = convolve(np.multiply(dI[0], dI[0]), patch)
        M[1] = convolve(np.multiply(dI[0], dI[1]), patch) 
        M[2] = convolve(np.multiply(dI[1], dI[1]), patch)

    return M.astype(float) 

def immatches(img, kpts_q, kpts_r, matches, color=(0,255,0)):
    kpts_q = swapXY(kpts_q[matches[0]])
    kpts_r = swapXY(kpts_r[matches[1]])
    return imlines(img, zip(kpts_q, kpts_r), color=color)

def imlines(img, lines, color=(0,0,255), thickness=2):
    if len(img.shape) == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255 
        img = img.astype(np.uint8) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 

    for line in lines:
        img = cv2.line(img, tuple(line[0]), tuple(line[1]), color=color, thickness=thickness) 

    return img 

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    if len(img.shape) == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 

    for pt in pts:
        img = cv2.circle(img, (round(pt[1]), round(pt[0])), r, color, thickness)

    return img

def imgrad(img, filter='Sobel', alg=0):
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) if filter=='Sobel' \
        else np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Prewitt 
    
    #---------------------------------------------------------------------------------------
    # compute with cv2.filter2D
    #--------------------------------------------------------------------------------------- 
    if alg==0:
        dI = np.zeros((2, img.shape[0], img.shape[1])) 
        dI[0] = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        dI[1] = cv2.filter2D(img, ddepth=-1, kernel=kernel.T)
    
    #---------------------------------------------------------------------------------------
    # compute with scipy.ndimage.filters.convolve 
    #--------------------------------------------------------------------------------------- 
    elif alg==1:
        img = img.astype(float) 
        dI = np.zeros((2, img.shape[0], img.shape[1])) 
        dI[0] = convolve(img, kernel) 
        dI[1] = convolve(img, kernel.T) 
    
    #---------------------------------------------------------------------------------------
    # compute with np.roll 
    #--------------------------------------------------------------------------------------- 
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

def plt_imshow(img, name="img", size=(15,5), cmap='gray'):
    plt.subplots(figsize=size) 
    plt.imshow(img, cmap = cmap) 
    plt.title(name)
    plt.show() 

def imshow(img, name="img", save=False):
    if save: cv2.imwrite(f'../results/{name}.png', img) 

    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break
    cv2.destroyAllWindows()

def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def swapXY(P):
    return np.roll(P.T, P.T.shape[1]).T

#---------------------------------------------------------------------------------------------------------------
#  tests 
#---------------------------------------------------------------------------------------------------------------
num = 20 
def run_test_imgrad(filename):
    '''
        test imgrad 
    '''
    print('*'*num + 'test imgrad' + '*'*num) 

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
        imshow(IxIy, f'IxIy_alg{i}', save=True)   

def run_test_imstructure(filename):
    '''
    test imstructure
    '''
    print('*'*num + 'test imstructure' + '*'*num) 

    img = imread(filename) 

    j = 0
    dI = imgrad(img, alg=j).astype(float) 
    
    for i in range(5):
        t0 = time.time()
        M = imstructure(dI, patch_radius=1, alg=i)
        t1 = time.time() 
        print(f'\n alg = {i} took {t1-t0:.4f} sec') 

        Ms = np.vstack((M[0], M[1], M[2])) 
        imshow(Ms, f'IxxIxyIyy_alg{i}', save=True)  

def run_test_scores(filename):
    '''
    test scores 
    '''
    print('*'*num + 'test scores' + '*'*num) 

    img = imread(filename) 
    dI = imgrad(img).astype(float) 
    M = imstructure(dI, patch_radius=1) 

    print('*'*5 + 'Harris' + '*'*5) 
    for i in range(3):
        t0 = time.time() 
        H = Harris(M, alg=i)
        t1 = time.time() 
        print(f'\n Harris score (alg={i}) took: {t1-t0:.4f} sec')
        imshow(H, f'Harris_alg{i}', save=True) 

        H[H<0] = 0
        imshow(H, f'Harris_alg{i}_ignore_edges', save=True) 

    print('*'*5 + 'ShiTomashi' + '*'*5) 
    for i in range(4):
        t0 = time.time() 
        S= ShiTomsashi(M, alg=i)
        t1 = time.time() 
        print(f'\n ShiTomashi score (alg={i}) took: {t1-t0:.4f} sec')
        imshow(S, f'ShiTomashi_alg{i}', save=True)

        S[S<0] = 0
        imshow(S, f'ShiTomashi_alg{i}_ignore_edges', save=True) 

def run_test_selectKeyPoints(filename):
    '''
    test selectKeyPoints
    '''
    print('*'*num + 'test selectKeyPoints' + '*'*num) 

    if 1:
        #------------------------------------------------------------------------------
        # test wtih a random small sample image with calculated scores 
        #------------------------------------------------------------------------------
        print('*'*5+'test on a random small sample image with relevant scores'+'*'*5) 
        height = 7
        width = 9
        img = np.random.randint(0,255,(height, width)).astype(float) 
        dI = imgrad(img)
        M = imstructure(dI, patch_radius=1)  
        scores = Harris(M) 
        
        print(f'\n img{img.shape, img.dtype} = \n', img) 
        print(f'\n dI{dI.shape, dI.dtype} = \n', dI) 
        print(f'\n scores{scores.shape, scores.dtype} = \n', scores) 
        print()

        kpts = selectKeyPoints(scores, img.shape, num_kpts=3, non_max_suppression_radius=2, alg=1)
        print('kpts = \n', kpts) 

        c = np.zeros_like(img) 
        for i in range(kpts.shape[0]):
            y,x = kpts[i]
            c[y,x] = 1 
        
        print(f'\n the given img {img.shape} is \n', img)
        diff_r = (img.shape[0] - scores.shape[0])//2
        scores = impad(scores, [diff_r,diff_r])
        print(f'\n scores_pad{scores.shape} are: \n', scores)
        print(f'\n key points {c.shape} are: \n', c)
    
    if 1:
        #------------------------------------------------------------------------------
        # test on a random small sample image with "random" scores 
        #------------------------------------------------------------------------------
        print('-'*200)
        print('*'*5+'test on a random small sample image with irrelevant scores'+'*'*5)
        print('-'*200)

        print('(so that we can see what is happening more simply)')
        height = 7
        width = 9
        r = 1
        img = np.random.randint(0,255,(height, width)).astype(float)  
        scores = np.random.randint(0,255,(height-2*r, width-2*r)) 
        print(f'\n img{img.shape, img.dtype} = \n', img) 
        print(f'\n dI{dI.shape, dI.dtype} = \n', dI) 
        print(f'\n scores{scores.shape, scores.dtype} = \n', scores) 
        print()

        kpts = selectKeyPoints(scores, img.shape, num_kpts=3, non_max_suppression_radius=2, alg=1)
        print('kpts = \n', kpts) 

        c = np.zeros_like(img) 
        for i in range(kpts.shape[0]):
            y,x = kpts[i]
            c[y,x] = 1 
        
        print(f'\n the given img {img.shape} is \n', img)
        diff_r = (img.shape[0] - scores.shape[0])//2
        scores = impad(scores, [diff_r,diff_r])
        print(f'\n scores_pad{scores.shape} are: \n', scores)
        print(f'\n key points {c.shape} are: \n', c)

    if 1:
        #------------------------------------------------------------------------------
        # test on a real image 
        #------------------------------------------------------------------------------
        print('*'*5 + 'test on a real image' + '*'*5)
        img = imread(filename)
        #img = cv2.GaussianBlur(img, (3,3), 1)
        dI = imgrad(img, alg=1).astype(float) 
        M = imstructure(dI, patch_radius=1) 
        R = Harris(M) 

        imshow(R) 

        t0 = time.time() 
        kpts = selectKeyPoints(R, img.shape, num_kpts=200, non_max_suppression_radius=9, alg=0)  
        t1 = time.time()
        print(f'\n selectKeyPoints (imgsize={img.shape}) took {t1-t0:.3f} sec') 
        print('kpts = \n', kpts) 
        
        c = np.zeros_like(img) 
        for i in range(kpts.shape[0]):
            y,x = kpts[i]
            c[y,x] = 1 
        print(f'\n key points {c.shape} are: \n', c)

        img = impoints(img, kpts)
        imshow(img, f'kpts', save=True)  

def run_test_describeKeyPoints(filename):
    img = imread(filename)
    dI = imgrad(img).astype(float) 
    M = imstructure(dI, patch_radius=1) 
    R = Harris(M) 
    kpts = selectKeyPoints(R, img.shape, alg=0)  
    
    r = 9
    descriptors = describeKeyPoints(img, kpts, descriptor_radius=r)

    for i in range(descriptors.shape[0]):
        imshow(descriptors[i].reshape(2*r+1,2*r+1))


def run_all(filenames):

    img_b = imread(filenames[0]) 
    dI_b = imgrad(img_b, alg=0).astype(float) 
    M_b = imstructure(dI_b, patch_radius=1) 
    R_b = Harris(M_b) 
    kpts_b = selectKeyPoints(R_b, img_b.shape, non_max_suppression_radius=8, alg=0) 
    des_b = describeKeyPoints(img_b, kpts_b, descriptor_radius=9)

    for i in range(1,len(filenames)-1):
        img_q = imread(filenames[i])
        dI_q = imgrad(img_q, alg=0).astype(float) 
        M_q = imstructure(dI_q, patch_radius=1)
        R_q = Harris(M_q)
        kpts_q = selectKeyPoints(R_q, img_q.shape, non_max_suppression_radius=8, alg=0)
        des_q = describeKeyPoints(img_q, kpts_q, descriptor_radius=9) 

        matches = matchDescriptors(des_q, des_b, match_lambda=4, alg=2) 

        

        img = impoints(img_q, kpts_q, color=(0,255,0)) 
        img = impoints(img, kpts_b, color=(0,0,255)) 

        if 1:
            img = immatches(img, kpts_q, kpts_b, matches) 
        else:
            q = kpts_q[matches[0]]
            b = kpts_b[matches[1]] 
            img = imlines(img, zip(swapXY(q), swapXY(b)), color=(0,255,0)) 
        imshow(img) 
        
        img_b = img_q 
        kpts_b = kpts_q 
        des_b = des_q


def run_test_cdist():
    print('*'*5 + 'test on cdist' + '*'*5)

    n_dim = 5
    Ma = 4
    Mb = 3
    a = np.random.randint(0,10, (Ma, n_dim))
    b = np.random.randint(0,10, (Mb, n_dim)) 
    print(f'\n a{a.shape} = \n', a)
    print(f'\n b{b.shape} = \n', b) 

    for i in range(Ma):
        row = a[i].reshape(1,-1) - b 
        print(f'\n a{i} - b = \n', row) 

        print('\n *^2 = \n', row**2) 

        print(f'\n sum(*^2) = \n', np.sum(row**2, axis=-1, keepdims=False))

        print(f'\n sqrt(sum(*^2)) = \n', np.sqrt(np.sum(row**2, axis=-1, keepdims=False)))

    measure = 'euclidean'
    dist = cdist(a, b, measure) 
    print(f'\n dist{dist.shape, measure} = \n', dist) 

 
if __name__=="__main__":
    main() 
