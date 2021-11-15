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

    p1xM1 = skew(p1) @ M1
    p2xM2 = skew(p2) @ M2
    A = np.concatenate([p1xM1, p2xM2], axis=-2)
    tAA = np.transpose(A, axes=(0,2,1)) @ A
    U, S, V = np.linalg.svd(tAA)
    
    P_sol = V[:,-1,:] / V[:,-1,-1].reshape(-1,1)

    return P_sol 


N = 10 #;         % Number of 3-D points

# Test linear triangulation
#P = randn(4,N);  % Homogeneous coordinates of 3-D points
#P(3, :) = P(3, :) * 5 + 10;
#P(4, :) = 1;

P = np.random.randn(4, N)
P[2, :] = P[2, :] * 5 + 10 
P[3, :] = 1     

M1 =  np.array([
        [500, 0, 320, 0],
        [0, 500, 240, 0],
        [0, 0, 1, 0]])

M2 =  np.array([
        [500, 0, 320, -100,],
        [0, 500, 240, 0],
        [0, 0, 1, 0]]) 
				
p1 = M1 @ P    #% Image (i.e., projected) points
p2 = M2 @ P

P_est = linearTriangulation(p1,p2,M1,M2)

print('P_est-P=\n')
print(P_est-P.T)
