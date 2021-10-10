import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import glob
from pdb import set_trace as db

def txt2array(txt, sep=' '):
    return np.fromstring(txt, dtype=float, sep=sep)

def imshow(img, name="img"):
    cv2.imshow(name, img)

    while True:
        if cv2.waitKey(1)==ord('q'):
            break 
    cv2.destroyAllWindows()

def add_points(img, ps, r=5, color=(0,0,255), thickness=-1):
    for p in ps:
        img = cv2.circle(img, (round(p[0]), round(p[1])), r, color, thickness)
    return img 

def load_data():
    K = open("../data/K.txt").read()
    p_W_corners = open('../data/p_W_corners.txt').read().split('\n')[:-1] 
    detected_corners = open('../data/detected_corners.txt').read() 
    filenames = glob.glob('../data/images_undistorted/*.jpg') 

    K = txt2array(K).reshape(3,3) 
    p_W_corners = np.array(list(txt2array(v, sep=",").tolist() for v in p_W_corners))
    detected_corners = txt2array(detected_corners).reshape(-1,12,2)
    filenames = sorted(filenames) 

    return K, p_W_corners, detected_corners, filenames

def homogeneous(P):
    return np.concatenate([P, np.ones((1,len(P))).T],axis=1)

def get_Q(xy1, XYZ1):
    Q = np.zeros((2*len(XYZ1), 12), dtype=float)
    Q[0::2,0:4] = XYZ1[:,:]
    Q[1::2,4:8] = XYZ1[:,:]
    Q[0::2,8:] = -xy1[:,0].reshape(12,1)*XYZ1[:,:]
    Q[1::2,8:] = -xy1[:,1].reshape(12,1)*XYZ1[:,:]
    return Q 

def get_M(Q):
    U, S, V = np.linalg.svd(Q) 

    M = V[-1]; 
    M = M if M[-1] > 0 else - M 
    #print(M) 

    M = M.reshape(3,4) 
    #print(M) 

    return M 

def get_Rt(M):
    R = M[:,:3]
    t = M[:,-1] 
    
    Ur,Sr,Vr = np.linalg.svd(R) 
    R_ = Ur @ Vr.T 

    alpha = np.linalg.norm(R_)/np.linalg.norm(R)
    t_ = alpha * t
    
    Rt = np.concatenate([R_, t_.reshape(1,-1).T], axis=1)

    if 0:
        print(np.linalg.det(R)) # 9.99459119111326e-06
        print(np.linalg.det(R_)) # 1.0000000000000002
        print(t) # [-0.39499907 -0.29955923  0.86766805]
        print(t_) # [-18.33628351 -13.90586292  40.27808817]
        print(Rt)
   
    return Rt

def estimatePoseDLT(p, P, K):
    p = np.round(p) 
    uv1 = homogeneous(p) 
    xy1 = (np.linalg.inv(K) @ uv1.T).T
    XYZ1 = homogeneous(P) 
    Q = get_Q(xy1, XYZ1) 
    M = get_M(Q) 
    Rt = get_Rt(M) 
    return Rt

def reprojectPoints(P, M, K):
    Ph = homogeneous(P) 
    p = K @ M @ Ph.T
    p = p / p[2]
    return p[:-1].T 

def main():
    K, p_W_corners, detected_corners, filenames = load_data() 
    p0 = detected_corners[0]

    M = estimatePoseDLT(p0, p_W_corners, K) 
    p = reprojectPoints(p_W_corners, M, K) 
    print(p0) 
    print(p) 


    img = cv2.imread(filenames[0])
    img = add_points(img, detected_corners[0])
    #img = add_points(img, p) 
    imshow(img) 

    #db() 

if __name__=="__main__":
    main() 
