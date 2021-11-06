import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import glob 

def txt2array(txt, sep=' '):
    return np.fromstring(txt, dtype=float, sep=sep)

def load_data():
    K = open("../data/K.txt").read()
    K = txt2array(K).reshape(3,3)

    p_W_corners = open('../data/p_W_corners.txt').read().split('\n')[:-1] 
    p_W_corners = np.array(list(txt2array(v, sep=",").tolist() for v in p_W_corners)) * 0.01

    detected_corners = open('../data/detected_corners.txt').read() 
    detected_corners = txt2array(detected_corners).reshape(-1,12,2)

    filenames = glob.glob('../data/images_undistorted/*.jpg') 
    filenames = sorted(filenames) 

    return K, p_W_corners, detected_corners, filenames

def imshow(img, name="img"):
    cv2.imshow(name, img) 

    while True:
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows() 

def add_points(img, pts, r=5, color=(0,0,255), thickness=-1):
    for pt in pts:
        img = cv2.circle(img, (round(pt[0]), round(pt[1])), r, color, thickness)
    return img 

def add_axis(img, pts):
    x0 = round(pts[0,0])
    y0 = round(pts[0,1]) 
    colors = [(0,0,255), (0,255,0), (255,0,0)]
    for i in [1, 2, 3]:
        xi = round(pts[i,0]); yi = round(pts[i,1])
        img = cv2.line(img, (x0, y0), (xi, yi), colors[i-1], 2)
    return img 

def homogeneous(P):
    return np.concatenate([P, np.ones((1,len(P))).T],axis=1)

def DirectLinearTransformationQ(xyh, XYZh):
    Q = np.zeros((2*len(XYZh), 12), dtype=float)
    Q[0::2,0:4] = XYZh[:,:]
    Q[1::2,4:8] = XYZh[:,:]
    Q[0::2,8:] = -xyh[:,0].reshape(12,1)*XYZh[:,:]
    Q[1::2,8:] = -xyh[:,1].reshape(12,1)*XYZh[:,:]
    return Q 

def Q2matM(Q):
    U, S, V = np.linalg.svd(Q) 

    M = V[-1] # this part can be modified 
    M = M if M[-1] > 0 else - M 
    M = M.reshape(3,4) 

    return M 

def M2Rt(M):
    R = M[:,:3]
    t = M[:,-1] 
    
    Ur,Sr,Vr = np.linalg.svd(R) 

    R_ = Ur @ Vr

    alpha = np.linalg.norm(R_)/np.linalg.norm(R) # R_ = alpha * R 

    t_ = alpha * t
    
    Rt = np.concatenate([R_, t_.reshape(1,-1).T], axis=1)

    if 0:
        print(np.linalg.det(R))     # 9.99459119111326e-06
        print(np.linalg.det(R_))    # 1.0000000000000002
        print(t)                    # [-0.39499907 -0.29955923  0.86766805]
        print(t_)                   # [-18.33628351 -13.90586292  40.27808817]
        print(Rt)
   
    return Rt

def estimatePoseDLT(p, P, K):
    uvh = homogeneous(p)
    xyh = (np.linalg.inv(K) @ uvh.T).T
    XYZh = homogeneous(P) 
    Q = DirectLinearTransformationQ(xyh, XYZh)
    M = Q2matM(Q) 
    Rt = M2Rt(M)
    return Rt 

def reprojectPoints(P, Rt, K):
    Ph = homogeneous(P) 
    p = K @ Rt @ Ph.T 
    p = p / p[2] 
    return p[:-1].T 

def plotTrajectory3D(M, P): 
    Rc = np.linalag.inv(M[:,:3])
    tc = - Rc @ M[:, 3]

    l = 0.1
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5)
    ax.quiver(tc[0], tc[1], tc[2], Rc[0,0], Rc[0,1], Rc[0,2], length=l, normalize=True, color="red", label="X")
    ax.quiver(tc[0], tc[1], tc[2], Rc[1,0], Rc[1,1], Rc[1,2], length=l, normalize=True, color="green", label="Y")
    ax.quiver(tc[0], tc[1], tc[2], Rc[2,0], Rc[2,1], Rc[2,2], length=l, normalize=True, color="blue", label="Z") 
    ax.scatter(P[:,0], P[:,1], P[:,2])

    plt.show()

def movieReprojection(filenames, detected_corners, p_W_corners, K):
    for i in range(len(filenames)):
        Rt = estimatePoseDLT(detected_corners[i], p_W_corners, K)
        p = reprojectPoints(p_W_corners, Rt, K)
        img = cv2.imread(filenames[i])
        img = add_points(img, p, r=2, color=(0,255,0))
        imshow(img)

def movie(filenames, detected_corners, P, K):
    for i in range(len(filenames)):       
        plt.subplots(figsize=(16, 9))

        M = estimatePoseDLT(detected_corners[i], P, K)  
        Rc = np.linalg.inv(M[:,:3])
        tc = - Rc @ M[:,3] 

        ax1 = plt.subplot(1,2,1)
        ax1.set_title("reprojection") 

        img = cv2.imread(filenames[i]) 
        img = add_points(img, reprojectPoints(P, M, K), r=2, color=(0,255,0)) 
        img = add_axis(img, reprojectPoints(axis_pts(), M, K))
        
        ax1.imshow(img)

        ax2 = plt.subplot(1,2,2, projection='3d')
        ax2.set_title("trajectory") 
        ax2.set_xlim(-0.5,0.5); ax2.set_ylim(-0.5,0.5); ax2.set_zlim(-0.5,0.5)

        ax2.scatter(P[:,0], P[:,1], P[:,2])

        l = 0.05

        ax2.quiver(tc[0], tc[1], tc[2], Rc[0,0], Rc[0,1], Rc[0,2], color="red", length=l, normalize=True, label="X")
        ax2.quiver(tc[0], tc[1], tc[2], Rc[1,0], Rc[1,1], Rc[1,2], color="green", length=l, normalize=True, label="Y")
        ax2.quiver(tc[0], tc[1], tc[2], Rc[2,0], Rc[2,1], Rc[2,2], color="blue", length=l, normalize=True, label="Z") 

        ax2.quiver(0, 0, 0, 1, 0, 0, length=l, normalize=True, label="x-axis", color="red")
        ax2.quiver(0, 0, 0, 0, 1, 0, length=l, normalize=True, label="y-axis", color="green")
        ax2.quiver(0, 0, 0, 0, 0, 1, length=l, normalize=True, label="z-axis", color="blue")

        ax2.view_init(-30, -100) 

        plt.show() 

def axis_pts():
    pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=float) 
    return pts * 0.01 

def main():
    K, p_W_corners, detected_corners, filenames = load_data() 

    q = detected_corners[0]

    M = estimatePoseDLT(q, p_W_corners, K) 
    p = reprojectPoints(p_W_corners, M, K) 
    #print(q); print(p) 

    img = cv2.imread(filenames[0])
    img = add_points(img, q, r=2, color=(0,0,255))
    img = add_points(img, p, r=2, color=(0,255,0)) 
    img = add_axis(img, reprojectPoints(axis_pts(), M, K)) # rgb = xyz here, but bgr = xyz in movie. why? 
    imshow(img) 

    k = 7
    movie(filenames[::k], detected_corners[::k], p_W_corners, K)
    
    #movieReprojection(filenames, detected_corners, p_W_corners, K) 
    
if __name__=="__main__":
    main() 
