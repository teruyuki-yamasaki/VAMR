import numpy as np 

def skew(p):
    op = np.array([
        [[0,0,0],[0,0,1],[0,-1,0]],
        [[0,0,-1],[0,0,0],[1,0,0]],
        [[0,1,0],[-1,0,0],[0,0,0]]
    ])

    px = np.transpose(op @ p, axes=(0,2,1)).T
    px = np.concatenate(px, axis=-1).reshape(-1,3,3)
    px = np.transpose(px, axes=(0,2,1)) 

    return px

def linearTriangulation(p1, p2, M1, M2):
    # LINEARTRIANGULATION  Linear Triangulation
    # Input:
    #  - p1(N, 3): homogeneous coordinates of points in image 1
    #  - p2(N, 3): homogeneous coordinates of points in image 2
    #  - M1(3,4): projection matrix corresponding to first image
    #  - M2(3,4): projection matrix corresponding to second image
    #
    # Output:
    #  - P(N, 4): homogeneous coordinates of 3-D points

    #TODO: create skew symmetric matrices of p1 and p2 
    p1x = skew(p1) 
    p2x = skew(p2) 

    #TODO: create the matrix 
    A = np.concatenate([p1x@M1, p2x@M2], axis=-2)

    #TODO: solve the linear system of equations AX=0 
    # by using the Singular Value Decomposition
    tA = np.transpose(A, axes=(0,2,1))
    tAA = tA @ A
    U, S, V = np.linalg.svd(tAA)
    P_sol = V[:,-1,:] / V[:,-1,-1].reshape(-1,1)

    return P_sol 
