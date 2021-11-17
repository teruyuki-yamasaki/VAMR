import numpy as np 

def selectKeyPoints(scores, num_kpts=200, non_max_suppression_r=1):
    # Given corner detection scores, select key points 
    # 
    # Input:
    #   - scores(height,width): corner detection scores 
    #   - num_kpts: the number of key points, 200 for default 
    #   - non_max_suppression_r: the patch radius for non max suppression of pixels surrounding local maxima 
    #
    # Output:
    #   - R(3,3,2) : the two possible rotations
    #   - u3(3,1)   : a vector with the translation information

    r = non_max_suppression_r
    scores_temp = impad(scores, [r,r]) 
    width = scores_temp.shape[1] 

    kpts = np.zeros((num_kpts, 2),dtype=int) 

    for i in range(num_kpts):
        # get the index of the pixel that gives the largest score among the remaining pixels 
        kp = np.argmax(scores_temp.flatten()) 

        # convert the above id into the pixel coordinate (y,x) in scores_temp     
        kp = np.array([kp//width, kp%width]) - r   

        # save the coordinate as the i-th key point's position 
        kpts[i, :] = kp

        # execute non maximum supresssion around the key point 
        scores_temp[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1] = np.zeros((2*r+1,2*r+1)) 
    
    return kpts 

def describeKeyPoints(img, kpts, descriptor_r=8):
    # Given an image and its key positions, create descriptors of the key points 
    # 
    # Input:
    #   - img(height,width): image in gray scale 
    #   - kpts(N, 2): pixel coordinates (y,x) of the key points 
    #   - discriptor_r: the patch radius for key point description 
    #
    # Output:
    #   - descriptor(N, 2*discriptor+1): discriptor 
    #   

    r = descriptor_r
    img_pad = impad(img, [r, r]) 

    descriptor = np.zeros((kpts.shape[0], (2*r+1)**2), dtype=np.uint8) 
    
    for i, kp in enumerate(kpts):
        descriptor[i, :] = img_pad[kp[0]:kp[0]+2*r+1, kp[1]:kp[1]+2*r+1].flatten()

    return descriptor

def matchDescriptors(query, base, match_lambda=4):
    # Given query discriptors and database discriptors, find discriptor correspondances 
    # 
    # Input:
    #   - query(Q, 2*r+1): query discriptors 
    #   - base(D, 2*r+1): database discriptors 
    #   - match_lambda: the const for threshold 
    #
    # Output:
    #   - descriptor(N, 2*discriptor+1): discriptor 
    #   

    # crate a look-at table of matches between discriptors of query and base 
    matchMat = np.zeros((query.shape[0], base.shape[0]), dtype=float)   

    for i in range(query.shape[0]):
        # compute distances between the i-th discriptor in query and all the discriptors in base 
        dist = np.sqrt( np.sum((query[i].reshape(1,-1) - base)**2, axis=1) ) 

        # regard the i-th discriptor in query as correspondence with a base discriptor
        # if their distance is the smallest 
        dist[dist!=np.min(dist)] = 0 
        matchMat[i, :] = dist 
        # are we assuming that the exact correspondance does not occur?
    
    # set a constraint that reduces unrealistic matches 
    min_nonzero_dist = np.min(matchMat[matchMat!=0]) # X[X!=0] extracts nonzero elements of X
    matchMat[matchMat >= min_nonzero_dist*match_lambda] = 0

    # remove double matches
    _, matchId = np.unique(matchMat, return_index=True)

    return matchMat, matchId[1:]
