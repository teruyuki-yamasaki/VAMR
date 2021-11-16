def decomposeEssentialMatrix(E):
    # Given an essential matrix, compute the camera motion, i.e.,  R and T such
    # that E ~ T_x R
    # 
    # Input:
    #   - E(3,3) : Essential matrix
    #
    # Output:
    #   - R(3,3,2) : the two possible rotations
    #   - u3(3,1)   : a vector with the translation information

    U, S, V = np.linalg.svd(E) 

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1] 
    ])

    R = np.zeros((3,3,2))
    R[:,:,0] = U @ W   @ V 
    R[:,:,1] = U @ W.T @ V 

    u = U[:,-1] 

    if 0:
        Tx = skew(u)
        E_ = Tx @ R 
        print(E_, E) 

    return R, u

def transformationMat(R, T):
    return np.hstack((R, T.reshape(1,3).T))

def disambiguateRelativePose(Rots, u3, points0_h, points1_h, K1, K2):
    # DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
    # four possible configurations) by returning the one that yields points
    # lying in front of the image plane (with positive depth).
    #
    # Arguments:
    #   Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
    #   u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
    #   p1   -  3xN homogeneous coordinates of point correspondences in image 1
    #   p2   -  3xN homogeneous coordinates of point correspondences in image 2
    #   K1   -  3x3 calibration matrix for camera 1
    #   K2   -  3x3 calibration matrix for camera 2
    #
    # Returns:
    #   R -  3x3 the correct rotation matrix
    #   T -  3x1 the correct translation vector
    #
    #   where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
    #   from the world coordinate system (identical to the coordinate system of camera 1)
    #   to camera 2.
    #

    M1 = K1 @ np.eye(3, 4) 
    M2 = K2 @ transformationMat(R, u3)  
    pass 

run_sfm(estimateEssentialMatrix, decomposeEssentialMatrix, disambiguateRelativePose) 
