import numpy as np 
from linalgs import krons, vec, unvec

def solveSystem(Q):
    #TODO: solve the linear system of equations Q @ vec(F)=0 
    # by using the Singular Value Decomposition

    # Singular Value Decomposition 
    U, S, V = np.linalg.svd(Q.T @ Q) 
    
    # choose the column of V that corresponds to the smallest eigenvalue, that minimizes
    vecF = V[-1]

    F = unvec(vecF) 

    Uf, Sf, Vf = np.linalg.svd(F) 
    Sf[-1] = 0 
    Sf = np.diag(Sf)
    Ff = Uf @ Sf @ Vf 

    if 0:
        print('Q.shape=', Q.shape) 
    
    if 0:
        print('U=\n', U)
        print('S=', S) 
        print('V=\n', V)
    
    if 0:
        #print('F=\n', F)
        print('\n detF=', np.linalg.det(F))
        print()
    
    if 0:
        print('\n Uf = \n', Uf)
        print('\n Sf = \n', Sf)
        print('\n Vf = \n', Vf) 
    
    if 0:
        #print('\n Ff = \n', Ff)
        print('\n detFf=', np.linalg.det(F))
        print()  
    
    return Ff

def fundamentalEightPoint(p1, p2):
    # fundamentalEightPoint  The 8-point algorithm for the estimation of the fundamental matrix F
    #
    # The eight-point algorithm for the fundamental matrix with a posteriori
    # enforcement of the singularity constraint (det(F)=0).
    # Does not include data normalization.
    #
    # Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.
    #
    # Input: point correspondences
    #  - p1(3,N): homogeneous coordinates of 2-D points in image 1
    #  - p2(3,N): homogeneous coordinates of 2-D points in image 2
    #
    # Output:
    #  - F(3,3) : fundamental matrix

    Q = krons(p1, p2) 

    F = solveSystem(Q) 

    return F

def normalizeMat(p):
    mu = np.mean(p[:2,:], axis=1, keepdims=False)
    sigma = np.mean(np.sum(np.square(p[:2,:] - mu.reshape(2,-1)), axis=0)) ** 0.5

    s = 2**0.5 / (sigma + 1e-8)

    T = np.array([
        [s, 0, -s*mu[0]],
        [0, s, -s*mu[1]],
        [0, 0, 1]])
    
    return T 


def fundamentalEightPoint_normalized(p1, p2):
    # estimateEssentialMatrix_normalized: estimates the essential matrix
    # given matching point coordinates, and the camera calibration K
    #
    # Input: point correspondences
    #  - p1(3,N): homogeneous coordinates of 2-D points in image 1
    #  - p2(3,N): homogeneous coordinates of 2-D points in image 2
    #
    # Output:
    #  - F(3,3) : fundamental matrix

    T1 = normalizeMat(p1)
    T2 = normalizeMat(p2) 
    p1 = T1@p1
    p2 = T2@p2

    Q = krons(p1, p2) 

    F = solveSystem(Q) 

    F = T2.T @ F @ T1

    return F 
