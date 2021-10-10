import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import glob

def txt2array(txt, sep=' '):
    return np.fromstring(txt, dtype=float, sep=sep)

def imshow(img, name="img"):
    cv2.imshow(name, img)

    while True:
        if cv2.waitKey(1)==ord('q'):
            break 
    cv2.destroyAllWindows()

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

def homogenous(P):
    return np.concatenate([P, np.ones((1,len(P))).T],axis=1)

def get_Q(xy, XYZ):
    XYZ1 = homogenous(XYZ) 

    Q = np.zeros((2*len(XYZ), 12), dtype=float)
    Q[0::2,0:4] = XYZ1[:,:]
    Q[1::2,4:8] = XYZ1[:,:]
    Q[0::2,8:] = -xy[:,0].reshape(12,1)*XYZ1[:,:]
    Q[1::2,8:] = -xy[:,1].reshape(12,1)*XYZ1[:,:]

    return Q 

def main():
    K, p_W_corners, detected_corners, filenames = load_data() 

    if 0:
        print(K)
        print(p_W_corners)
        print(detected_corners) 
        print(filenames) 
    
    Q = get_Q(detected_corners[0], p_W_corners) 
    print(Q) 


if __name__=="__main__":
    main() 
