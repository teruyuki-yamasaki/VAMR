import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import glob 

def main():

    args = {
        'K': '../data/K.txt',
        'p_W_corners': '../data/p_W_corners.txt',
        'detected_corners': '../data/detected_corners.txt',
        'filenames': '../data/images_undistorted/*.jpg',
        'arrow_size': 1
    } 
    
    
    K = txt2array(args['K']).reshape(3,3)
    p_W_corners = txt2array(args['p_W_corners']).reshape(-1,3).transpose(1,0) 
    detected_corners = txt2array(args['detected_corners']).reshape(-1,12,2).transpose(0,2,1)
    filenames = sorted(glob.glob(args['filenames']))
    
    if 1:
        print(f'\n K (3,3) = \n', K)
        print(f'\n p_W_corners {p_W_corners.shape} =  \n', p_W_corners) 
        print(f'\n detected_corners {detected_corners.shape} =  \n', detected_corners) 
        print('\n filenames = ')
        for i in range(3): print(filenames[i]) 
        print('...')
    
    if 1:
        Pwh = homogeneous(p_W_corners) 
        print('\n Pwh = \n', Pwh) 

        ph = homogeneous(detected_corners[0]) 
        print('\n ph = \n', ph) 

    if 1:
        q = detected_corners[0]
        M = estimatePoseDLT(q, p_W_corners, K) 
        p = reprojectPoints(p_W_corners, M, K)

        if 1: 
            print('\n q = \n', q)
            print('\n p = \n', p)  
    
    if 1:
        img = imread(filenames[0])
        img = imdots(img, q, r=2, color=(0,0,255))
        img = imdots(img, p, r=2, color=(0,255,0)) 

        axes = poseAxes(args['arrow_size'])
        axes = reprojectPoints(axes, M, K)

        if 1:
            print('\n axes before reprojection = \n', poseAxes(args['arrow_size']))
            print('\n axes after reprojection = \n', axes)  

        img = imaxes(img, axes) # rgb = xyz here, but bgr = xyz in movie. why? 
        imshow(img) 
    
    if 0:
        k = 7
        movieReprojection(filenames[::k], detected_corners[::k], p_W_corners, K) 
    
    if 1:
        plotTrajectory3D(M, p_W_corners)  
    
    if 1:
        k = 7
        movie(filenames[::k], detected_corners[::k], p_W_corners, K)

def movie(filenames, detected_corners, P, K):
    for i in range(len(filenames)):

        # estimate the camera pose 
        M = estimatePoseDLT(detected_corners[i], P, K)  
        Rc = np.linalg.inv(M[:,:-1])
        tc = - Rc @ M[:,-1] 
        
        #tc *= 0.01 
        #P *= 0.01

        # visualize 
        plt.subplots(figsize=(16, 9))

        # fig.1: corners on each image 
        ax1 = plt.subplot(1,2,1)
        ax1.set_title("reprojection") 

        img = imread(filenames[i]) 
        img = imdots(img, reprojectPoints(P, M, K), r=2, color=(0,255,0)) 
        img = imaxes(img, reprojectPoints(poseAxes(0.01), M, K))

        ax1.imshow(img)

        # fig.2: a 3d view of the trajectory of the estimated camera pose 
        ax2 = plt.subplot(1,2,2, projection='3d')
        ax2.set_title("trajectory") 
        ax2.set_xlim(-0.5,0.5); ax2.set_ylim(-0.5,0.5); ax2.set_zlim(-0.5,0.5)

        l = 0.05

        # the camera poses 
        ax2.quiver(tc[0], tc[1], tc[2], Rc[0,0], Rc[0,1], Rc[0,2], color="red", length=l, normalize=True, label="X")
        ax2.quiver(tc[0], tc[1], tc[2], Rc[1,0], Rc[1,1], Rc[1,2], color="green", length=l, normalize=True, label="Y")
        ax2.quiver(tc[0], tc[1], tc[2], Rc[2,0], Rc[2,1], Rc[2,2], color="blue", length=l, normalize=True, label="Z") 

        # the axes of the 3D world coordinate system 
        ax2.quiver(0, 0, 0, 1, 0, 0, length=l, normalize=True, label="x-axis", color="red")
        ax2.quiver(0, 0, 0, 0, 1, 0, length=l, normalize=True, label="y-axis", color="green")
        ax2.quiver(0, 0, 0, 0, 0, 1, length=l, normalize=True, label="z-axis", color="blue")

        # the reference points 
        ax2.scatter(P[0], P[1], P[2])

        # set an appropriate viewpoint 
        ax2.view_init(-30, -100) 

        plt.show() 

def plotTrajectory3D(M, P): 
    Rc = np.linalg.inv(M[:,:-1])
    tc = - Rc @ M[:, -1] 
    
    tc *= 0.01 
    P *= 0.01

    l = 0.1
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5)

    # the camera pose 
    ax.quiver(tc[0], tc[1], tc[2], Rc[0,0], Rc[0,1], Rc[0,2], length=l, normalize=True, color="red", label="X")
    ax.quiver(tc[0], tc[1], tc[2], Rc[1,0], Rc[1,1], Rc[1,2], length=l, normalize=True, color="green", label="Y")
    ax.quiver(tc[0], tc[1], tc[2], Rc[2,0], Rc[2,1], Rc[2,2], length=l, normalize=True, color="blue", label="Z") 

    # the reference points 
    ax.scatter(P[0], P[1], P[2])

    # the axes of the 3D world coordinate system 
    ax.quiver(0, 0, 0, 1, 0, 0, length=l, normalize=True, label="x-axis", color="red")
    ax.quiver(0, 0, 0, 0, 1, 0, length=l, normalize=True, label="y-axis", color="green")
    ax.quiver(0, 0, 0, 0, 0, 1, length=l, normalize=True, label="z-axis", color="blue")
    
    # set an appropriate viewpoint q
    ax.view_init(-30, -100) 

    plt.show()

def movieReprojection(filenames, detected_corners, p_W_corners, K):
    for i in range(len(filenames)):
        Rt = estimatePoseDLT(detected_corners[i], p_W_corners, K)
        p = reprojectPoints(p_W_corners, Rt, K)
        img = imread(filenames[i])
        img = imdots(img, p, r=2, color=(0,255,0))
        axes = reprojectPoints(poseAxes(), Rt, K)
        img = imaxes(img, axes) 
        imshow(img)


def reprojectPoints(P, Rt, K): 
    '''
    Input:
        P(3, numPoints): 3D world coords of points 

    Output:
        p(2, numPoints): 2D pixel coords of projected points 
    '''

    p = K @ Rt @ homogeneous(P)

    p = p / p[2] 

    return p[:-1]

def estimatePoseDLT(p, P, K):
    '''Implement the steps of the DLT algorithm to solve for the projectino matrix M = [R|t] 

    Input:
        - p(2, numPoints): 2D pixel coords of detected points 
        - P(3, numPoints): 3D world coords of the reference points 
        - K(3,3): the camera matrix 
    '''

    xyh = np.linalg.inv(K) @ homogeneous(p) # !the pixel coords to the normalized coords (or the calibrated coords) 
    XYZh = homogeneous(P) 
    Q = DirectLinearTransformationSLE(xyh, XYZh) 
    M = solveSVD(Q).reshape(3,4) 
    Rt = extractRt(M)
    return Rt 

def extractRt(M):
    '''extract the projection matrix [R|t] from M with correct scale 
    '''
    # Make sure that the z component of the resocvered transitino Tz is positive: 
    M = M if M[-1,-1] > 0 else -M

    # Extraxt rotation and transclation before scaling 
    r = M[:,:-1]    # Rotation Matrix before scaling 
    t = M[:,-1]     # Translation before scaling 
    
    # Extract a true rotation 
    Ur,Sr,Vr = np.linalg.svd(r) # first decompose the estimated rotation r using the SVD 
    R = Ur @ Vr                 # then force all the eigenvalues to be 1 

    # Recover the unknown scaling factor alpha = ||R||/||r|| 
    alpha = np.linalg.norm(R)/np.linalg.norm(r) 

    # Recover the final projection matrix 
    T = alpha * t
    RT = np.concatenate([R, T.reshape(1,-1).T], axis=1)

    if 0:
        print('\n det(r) = ', np.linalg.det(r))     # 9.99459119111326e-06
        print('\n det(R) = ', np.linalg.det(R))     # 1.0000000000000002
        print('\n T = \n', T)                       # [-18.33628351 -13.90586292  40.27808817]
        print('\n R = \n', R)                       # [[ 0.61754966  0.12886497 -0.77590349],[-0.34974877  0.92858143 -0.12414639],[ 0.70449145  0.34803786  0.61851552]]
        print('\n RT = \n', RT)
   
    return RT


def solveSVD(A):
    ''' solve a system of linear equations that is over-determined:
            AX = 0
        by using the Singluar Value Decomposition (SVD).

        We look for a solution that minimizes ||AX|| subject to the constraint ||X|| = 1.
        Since the solution of this problem is the eigen vector 
        corresponding to the smallest eigen value of tAA, 
        we only need to obtain the last column of V, s.t. U @ S @ V = svd(tAA) 
        where U and V are unitary matrices. 
        if S has its diagonal entries stored in descending order. 

        Note that in NumPy, S is given as a 1D vector of the eigenvalues stored in descending order. 
    '''
    U, S, V = np.linalg.svd(A)  

    M = V[-1]  

    return M 

def DirectLinearTransformationSLE(xyh, XYZh):
    ''' construct a system of linear equations (SLE) to determine the pose matrix 
        using the Direct Linear Transformation algorithm 
        given detected corners and world corners positions 

    Input:
        - xyh(2+1, numPoints): homogenous coords of 2D pixcel points 
        - XYZh(3+1, numPoints): homogenous coords of 3D world points 

    Output
        - Q(2*numPoints, 12): the SLE matrix for DLT 
    
    '''
    numPoints = XYZh.shape[1] 
    Q = np.zeros((2*numPoints, 12), dtype=float)
    
    Q[0::2,0:4] = XYZh.T
    Q[1::2,4:8] = XYZh.T 
    Q[0::2,8:] = -xyh[0].reshape(12,1)*XYZh.T
    Q[1::2,8:] = -xyh[1].reshape(12,1)*XYZh.T 

    return Q 

def homogeneous(P):
    '''retrun homgenous coords of given points

    Innput: 
        P(dim, numPoints)
    
    Pitput:
        Ph(dim+1, numPoints) 
    '''
    return np.concatenate([P, np.ones(P.shape[-1]).reshape(1,-1)], axis=-2) 

def poseAxes(arrow_size=1):
    axes = np.array([
        [0, 0, 0],  # O 
        [1, 0, 0],  # ex 
        [0, 1, 0],  # ey
        [0, 0, 1]], # ez 
        dtype=np.uint8).T 
    return axes * arrow_size 

def imaxes(img, pts):
    '''visualize pose 

    Input:
        img 
        pts(2, 3): 3D world coords of relative origin, ex, ey, and ez 
    
    Output:
        img with the input pose visualized 
    '''
    x0 = round(pts[0,0])
    y0 = round(pts[1,0]) 
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    for i in [1, 2, 3]:
        xi = round(pts[0,i]); yi = round(pts[1,i])
        img = cv2.line(img, (x0, y0), (xi, yi), colors[i-1], 2)

    return img 

def imdots(img, pts, r=5, color=(0,0,255), thickness=-1):
    for i in range(pts.shape[1]):
        img = cv2.circle(img, (round(pts[0,i]), round(pts[1,i])), r, color, thickness)
    return img 

def imshow(img, name="img"):
    cv2.imshow(name, img) 

    while True:
        if cv2.waitKey(1)==ord('q'): break
        
    cv2.destroyAllWindows() 

def imread(filename):
    return cv2.imread(filename) 

def txt2array(filename, sep=' '):  
    txt = open(filename).read() 
    return np.fromstring(txt, dtype=float, sep=sep)
    
if __name__=="__main__":
    main() 
