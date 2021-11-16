import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

def main():    
    if 0:
        # Load data 
        p1 = txt2array('../data/matches0001.txt').reshape(2,-1)
        p2 = txt2array('../data/matches0002.txt').reshape(2,-1)
        img1 = imread('../data/0001.jpg')
        img2 = imread('../data/0002.jpg')
        immatches(img1, img2, p1, p2)
    
    if 1:
        run_test_linearTriangulation(linearTriangulation)
    
    if 0:
        run_test_linalgs(homogenous, vec, unvec, krons)
    
    if 1:
        run_test_8point(fundamentalEightPoint, fundamentalEightPoint_normalized)
    
    if 1:
        run_sfm(estimateEssentialMatrix, decomposeEssentialMatrix, disambiguateRelativePose) 

def run_sfm(estimateEssentialMatrix, decomposeEssentialMatrix, disambiguateRelativePose):

    # Load data sets 
    img1 = imread('../data/0001.jpg')
    img2 = imread('../data/0002.jpg')
    
    K = np.array([
            [1379.74, 0, 760.35],
            [0, 1382.08, 503.41],
            [0, 0, 1]]) 
    
    # Load outlier-free point correspondences
    p1 = txt2array('../data/matches0001.txt').reshape(2,-1)
    p2 = txt2array('../data/matches0002.txt').reshape(2,-1)

    p1 = homogenous(p1)
    p2 = homogenous(p2) 

    # Estimate the essential matrix E using the 8-point algorithm
    E = estimateEssentialMatrix(p1, p2, K, K)

    # Extract the relative camera positions (R,T) from the essential matrix
    # Obtain extrinsic parameters (R,t) from E
    [Rots, u3] = decomposeEssentialMatrix(E)
    # Disambiguate among the four possible configurations
    [R_C2_W, T_C2_W] = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

    # Triangulate a point cloud using the final transformation (R,T)
    M1 = K @ np.eye(3,4)
    M2 = K @ transformationMat(R_C2_W, T_C2_W)
    P = linearTriangulation(p1, p2, M1, M2)

    visualize(img1, img2, p1, p2, P, R_C2_W, T_C2_W)

def estimateEssentialMatrix(p1, p2, K1, K2):
    # estimateEssentialMatrix_normalized: estimates the essential matrix
    # given matching point coordinates, and the camera calibration K
    #
    # Input: point correspondences
    #  - p1(3,N): homogeneous coordinates of 2-D points in image 1
    #  - p2(3,N): homogeneous coordinates of 2-D points in image 2
    #  - K1(3,3): calibration matrix of camera 1
    #  - K2(3,3): calibration matrix of camera 2
    #
    # Output:
    #  - E(3,3) : fundamental matrix

    F = fundamentalEightPoint_normalized(p1, p2) 

    E = K2.T @ F @ K1 

    return E 

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

    u = U[:,-1] 

    R = np.zeros((3,3,2))
    R[:,:,0] = U @ W   @ V 
    R[:,:,1] = U @ W.T @ V 

    for i in range(2):
        if np.linalg.det(R[:,:,i]) < 0: R[:,:,i] *= -1 

    if 1:
        Tx = skews(u.reshape(-1,3)) 
        E_ = Tx @ R

        print('detR = %f' % np.linalg.det(R[:,:,0]))
        print('detR.T = %f' % np.linalg.det(R[:,:,1]))

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

    R0 = Rots[:,:,0]
    R1 = Rots[:,:,1]

    M1 = K1 @ np.eye(3, 4) 

    X2 = np.zeros((3,4,4))
    X2[:,:,0] = transformationMat(R0,  u3)  
    X2[:,:,1] = transformationMat(R1,  u3)
    X2[:,:,2] = transformationMat(R0, -u3) 
    X2[:,:,3] = transformationMat(R1, -u3) 

    for i in range(4):
        M2 = K2 @ X2[:,:,i]

        P = linearTriangulation(points0_h, points1_h, M1, M2)

        if np.all(P[2,:].flatten()>0):
            R_C2_W, T_C2_W = X2[:,:3,i], X2[:,3,i]

    return R_C2_W, T_C2_W 

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

    #Q = krons(p1, p2) 
    #F = solveSystem(Q) 
    F = fundamentalEightPoint(p1, p2)

    F = T2.T @ F @ T1

    return F 

def normalizeMat(p):
    # normaliation Matrix: 
    # given point coordinates
    #
    # Input: point correspondences
    #  - p(3,N): homogeneous coordinates of 2-D points in an image 
    #
    # Output:
    #  - T(3,3) : normalization operator 
    #               p[:,i] = s_j * (p[:,i] - pc) (i=1:n, s = 2**0.5 / sigma, pc:centroid)  

    # the centroid: by taking the mean of each coordinate 
    pc = np.mean(p[:2,:], axis=1, keepdims=True) 
    sigma = np.mean(np.sum(np.square(p[:2,:] - pc), axis=0)) ** 0.5

    s = 2**0.5 / (sigma + 1e-8)

    T = np.array([
        [s, 0, -s*pc[0,0]],
        [0, s, -s*pc[1,0]],
        [0, 0, 1]])
    
    return T 

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


def linearTriangulation(p1, p2, M1, M2):
    # LINEARTRIANGULATION  Linear Triangulation
    # Input:
    #  - p1(3, N): homogeneous coordinates of points in image 1
    #  - p2(3, N): homogeneous coordinates of points in image 2
    #  - M1(3,4): projection matrix corresponding to first image
    #  - M2(3,4): projection matrix corresponding to second image
    #
    # Output:
    #  - P(4, N): homogeneous coordinates of 3-D points

    #TODO: create skew symmetric matrices of p1 and p2 
    p1x = skews(p1.T) 
    p2x = skews(p2.T) 

    #TODO: create a matrix for the linear system of equations 
    A = np.concatenate([p1x@M1, p2x@M2], axis=-2)

    #TODO: solve the linear system of equations AX=0 
    # by using the Singular Value Decomposition
    tA = np.transpose(A, axes=(0,2,1))
    tAA = tA @ A
    U, S, V = np.linalg.svd(tAA)
    P_sol = V[:,-1,:] / V[:,-1,-1].reshape(-1,1)

    return P_sol.T

def skews(p):
    # Skew Symmetric Matrices
    # Input:
    #  - p(N, 3): homogeneous coordinates of points
    #
    # Output:
    #  - px(N, 3, 3): Skew Symmetric Matrix of points 

    op = np.array([
        [[0,0,0],[0,0,1],[0,-1,0]],
        [[0,0,-1],[0,0,0],[1,0,0]],
        [[0,1,0],[-1,0,0],[0,0,0]]
    ])

    px = np.transpose(op @ p.T, axes=(0,2,1)).T
    px = np.concatenate(px, axis=-1).reshape(-1,3,3)
    px = np.transpose(px, axes=(0,2,1)) 

    return px

def homogenous(p): 
    # returns homogenous coordinates of n-D points
    # given coordinates of n-D points  

    # Input:
    #   - p(n, N): coordinates of n-D points
    # 
    # Output:
    #   - ph(n+1, N): homogenous coordinates of n-D points 

    return np.vstack((p, np.ones(p.shape[1]))) 

def vec(F):
    # stack vec F's columns into a single column vector 

    # Input:
    #   - F(3,3): a 3x3 matrix 
    # 
    # Output:
    #   - vecF(9): a 9D vector 

    return F.T.flatten()

def unvec(F):
    # do the oposite operation of vec(F) 
    # decompose vec F into three columns 

    # Input:
    #   - F(9): a 9-D vector 
    # 
    # Output:
    #   - matF(3,3): a 3x3 matrix  

    return F.reshape(3,3).T 

def krons(p1, p2):
    # stack the Kronecker products of N 2-D homogenous coodinates into a matrix 

    # Input:
    #   - p1(3, N): 
    #   - p2(3, N):
    # 
    # Output:
    #   - Q(N, 9):

    N = p1.shape[1]
    Q = np.zeros((N, 9)) # dtype ?? 

    for i in range(N):
        Q[i] = np.kron(p1[:,i], p2[:,i]) 
    
    return Q 

# implags 
def immatches(img1, img2, p1, p2, size=(30,10)):
    # Display matched points
    plt.figure(figsize=size)

    plt.subplot(1,3,2)
    plt.imshow(img1)
    plt.plot(p1[0,:], p1[1,:], 'ys')
    plt.title('Image 1')

    plt.subplot(1,3,3)
    plt.imshow(img2);
    plt.plot(p2[0,:], p2[1,:], 'ys')
    plt.title('Image 2')

    plt.show() 

def txt2array(filename, sep='\n'):
    txt = open(filename).read() 
    return np.fromstring(txt, dtype=float, sep=sep)

def imread(filename):
    return cv2.imread(filename) 

def imshow(img, name='img', size=(15,5), cmap='gray'): #'viridis'
    plt.subplots(figsize=size) 
    plt.imshow(img, cmap = cmap) 
    plt.title(name)
    plt.show() 

def imcompare(img1, img2, size=(30,10)):
    plt.figure(figsize=size)

    plt.subplot(1,3,2)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(1,3,3)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.show() 

#tests 
######################################################
num = 20

def run_test_linalgs(homogenous, vec, unvec, krons):
    print('*'*num + 'Test linear algebreic operations' + '*'*num + '\n')  

    N = 4
    m = 8

    F = np.random.randint(1,m,(3,3)) 

    p1 = np.random.randint(1,m,(2,N))
    p2 = np.random.randint(1,m,(2,N)) 

    if 1: 
        print('*check the initital parameters*') 
        print('\n NumPoints=', N)
        print('\n F= \n', F)
        print('\n p1='); print(p1) 
        print('\n p2='); print(p2)
        print('\n') 
    
    if 1:
        print('*check vectrization and unvectrization*')
        print('\n flatten F={}'.format(F.flatten()))
        print('\n vec(F) = {}'.format(vec(F))) 
        print('\n unvec(vec(F)) = \n {}'.format(unvec(vec(F))))
        print('\n')

    p1 = homogenous(p1)
    p2 = homogenous(p2)

    if 1:
        print('*check homogenous*')
        print('\n p1='); print(p1) 
        print('\n p2='); print(p2)
        print('\n') 
    
    if 0: # check np.kron(a, b) 
        i = 1
        print('i = %d' % i)
        print('p2.T[:,i] @ F @ p1[:,i] = %d' % (p2[:,i].T @ F @ p1[:,i]))
        print('np.kron(p1[:,i], p2[:,i]).T @ vec(F)= %d \n' % (np.kron(p1[:,i], p2[:,i]).T @ vec(F)))

        print('np.kron(p1[:,i], p2[:,i]).T=', np.kron(p1[:,i], p2[:,i]).T)
        print('np.kron(p1, p2).T=\n', np.kron(p1, p2).T)
        print('\n')

    
    Q = krons(p1, p2) 

    print('*check krons*')
    print('\n Q ='); print(Q)

    print('\n Q @ vec(F) = ', Q @ vec(F)) 
    print('\n p2[:,i].T @ F @ p1[:,i] = ', [p2[:,i].T @ F @ p1[:,i] for i in range(p1.shape[1])])
    print('\n diff = ', Q @ vec(F) - np.array( [p2[:,i].T @ F @ p1[:,i] for i in range(p1.shape[1])]))

    print('\n')

def run_test_linearTriangulation(linearTriangulation):
    # Test linear triangulation

    print('*'*num + 'Test linearTriangulation' + '*'*num + '\n')  

    N = 10 # Number of 3-D points

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
                    
    p1 = M1 @ P    ## Image (i.e., projected) points
    p2 = M2 @ P

    P_est = linearTriangulation(p1,p2,M1,M2)

    print('P_est-P=\n')
    print(P_est-P)

    print('\n')

def sumSquaredEpipolarConstraints(F, x1, x2):
    # the sum of Squared Epipolar Constrains: compute the sum of Squared Epipolar Constrains
    #
    #   Input:
    #   - F(3,3): Fundamental matrix
    #   - p1(3,N): homogeneous coords of the observed points in image 1
    #   - p2(3,N): homogeneous coords of the observed points in image 2
    #
    #   Output:
    #   - cost: sum of squared distance from points to epipolar lines
    #           normalized by the number of point coordinates

    N = x1.shape[1]

    if 1: 
        tx2Fx1 = np.array([x2[:,i].T @ F @ x1[:,i] for i in range(N)])

    else:
        tx2Fx1 = krons(x1, x2) @ vec(F)

    cost = np.linalg.norm( tx2Fx1 ) / np.sqrt(N) 

    return cost 

def distPoint2EpipolarLine(F, p1, p2):
    # distPoint2EpipolarLine  Compute the point-to-epipolar-line distance
    #
    #   Input:
    #   - F(3,3): Fundamental matrix
    #   - p1(3,NumPoints): homogeneous coords of the observed points in image 1
    #   - p2(3,NumPoints): homogeneous coords of the observed points in image 2
    #
    #   Output:
    #   - cost: sum of squared distance from points to epipolar lines
    #           normalized by the number of point coordinates

    NumPoints = p1.shape[1] 

    points = np.hstack([p1, p2]) 
    epi_lines = np.hstack([F.T@p2, F@p1])

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2

    b = epi_lines * points 
    a = np.sum(b, axis=0)

    cost = np.sqrt( np.sum( a**2 / denom ) / NumPoints )

    if 0:
        print('\n p1= \n', p1)
        print('\n p2= \n', p2)
        print('\n homog_points=\n', points*1e-3)
        print('\n epi_lines=\n', epi_lines*1e-3)
        print('\n a=\n', a)
        print('\n cost=\n', cost) 
        print(denom.shape) # (80,)
        print(a.shape) # (80,)
        print(b.shape) # (3, 80)

    return cost 

def run_test_8point(fundamentalEightPoint, fundamentalEightPoint_normalized):
    print('*'*num + 'Test 8point' + '*'*num + '\n')  

    N = 40;         # Number of 3-D points
    X = np.random.randn(4,N);  # Homogeneous coordinates of 3-D points

    # Simulated scene with error-free correspondences
    X[2, :] = X[2, :] * 5 + 10;
    X[3, :] = 1;

    P1 = np.array([
            [500, 0, 320, 0],
            [0, 500, 240, 0],
            [0, 0, 1, 0]]);

    P2 = np.array([
            [500, 0, 320, -100],
            [0, 500, 240, 0],
            [0, 0, 1, 0]]);
                    
    x1 = P1 @ X;     # Image (i.e., projected) points
    x2 = P2 @ X;

    ## Fundamental matrix estimation via the 8-point algorithm

    # Estimate fundamental matrix
    # Call the 8-point algorithm on inputs x1,x2
    F = fundamentalEightPoint(x1,x2);

    # Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
    cost_algebraic = sumSquaredEpipolarConstraints(F, x1, x2)
    cost_dist_epi_line = distPoint2EpipolarLine(F, x1, x2);

    print('Noise-free correspondences\n');
    print('Algebraic error: %f \n' % cost_algebraic);
    print('Geometric error: %f px\n\n' % cost_dist_epi_line);

    ## Test with noise:

    sigma = 1e-1
    noisy_x1 = x1 + sigma * np.random.randn(x1.shape[0]).reshape(-1,1)
    noisy_x2 = x2 + sigma * np.random.randn(x2.shape[0]).reshape(-1,1)

    # Estimate fundamental matrix
    # Call the 8-point algorithm on noisy inputs x1,x2
    F = fundamentalEightPoint(noisy_x1,noisy_x2)

    # Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
    cost_algebraic = sumSquaredEpipolarConstraints(F, noisy_x1, noisy_x2) 
    cost_dist_epi_line = distPoint2EpipolarLine(F, noisy_x1, noisy_x2)

    print('Noisy correspondences (sigma=%f), with fundamentalEightPoint\n' % sigma)
    print('Algebraic error: %f \n' % cost_algebraic)
    print('Geometric error: %f px\n\n' % cost_dist_epi_line)

    ## Normalized 8-point algorithm
    # Call the normalized 8-point algorithm on inputs x1,x2
    Fn = fundamentalEightPoint_normalized(noisy_x1,noisy_x2)

    # Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
    cost_algebraic = sumSquaredEpipolarConstraints(Fn, noisy_x1, noisy_x2)
    cost_dist_epi_line = distPoint2EpipolarLine(Fn, noisy_x1, noisy_x2)

    print('Noisy correspondences (sigma=%f), with fundamentalEightPoint_normalized\n' % sigma)
    print('Algebraic error: %f \n' % cost_algebraic)
    print('Geometric error: %f px\n\n' % cost_dist_epi_line)

def visualize(img1, img2, p1, p2, P, R_C2_W, T_C2_W): 
    # Visualize the 3-D scene
    ax = plt.axes(projection='3d')

    # R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

    # P is a [4xN] matrix containing the triangulated point cloud (in homogeneous coordinates), 
    # given by the function linearTriangulation
    ax.plot3D(P[0,:], P[1,:], P[2,:], 'o')

    # Display camera pose

    #plotCoordinateFrame(np.eye(3), np.zeros((3,1)), 0.8);
    #ax.text(-0.1,-0.1,-0.1, "Cam=1", fontsize=10, color='k', FontWeight='bold')

    center_cam2_W = -R_C2_W.T @ T_C2_W
    #plotCoordinateFrame(R_C2_W.T, center_cam2_W, 0.8);
    #ax.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2',fontsize=10,color='k',FontWeight='bold')

    # Display matched points
    immatches(img1, img2, p1, p2) 


if __name__=='__main__':
    main() 
