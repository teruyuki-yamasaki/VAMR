import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import glob

def txt2array(txt, sep=' '):
    return np.fromstring(txt, dtype=float, sep=sep)

def imshow(img):
    cv2.imshow('img',img)
    while True:
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()

def add_corners(img, p, r=5, color=(0,0,255), thickness=-1):
    for x,y in zip(p[0], p[1]):
        img = cv2.circle(img, (round(x), round(y)), r, color, thickness) # -1: fill the circle
    return img

def add_cube(img, p, r=5, color=(0,0,255)):
    A = [0, 0, 1, 2, 4, 4, 5, 6, 0, 1, 2, 3]
    B = [1, 2, 3, 3, 5, 6, 7, 7, 4, 5, 6, 7]

    for i, j in zip(A, B):
        x0 = round(p[0, i]); y0 = round(p[1, i])
        x1 = round(p[0, j]); y1 = round(p[1, j])
        img = cv2.line(img, (x0, y0), (x1, y1), color, r)

    return img

def plotXYZ(XX, YY, ZZ): # ZZ = f(XX, YY)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(XX, YY, ZZ)
    plt.show()

def swapXY(P):
    P_ = np.zeros_like(P)
    P_[:] = P[:]
    P_[:,0] = P[:,1]
    P_[:,1] = P[:,0]
    return P_

def create_corner_dots(H=6, W=9, dx=0.04, dy=0.04):
    X = np.arange(H) * dx
    Y = np.arange(W) * dy

    XX, YY = np.meshgrid(X, Y)
    ZZ = XX*0 + YY*0

    XYZ = np.stack((XX, YY, ZZ), axis=-1).reshape(-1, 3)

    XYZ = swapXY(XYZ)

    return XYZ

def create_cube_dots(xy=(0,0), size=1, dx=0.04, dy=0.04):

    X = np.array([xy[0], xy[0]+size]) * dx
    Y = np.array([xy[1], xy[1]+size]) * dy

    XX, YY = np.meshgrid(X, Y)
    ZZ0 = XX*0 + YY*0
    ZZ1 = -0.5 * size * (np.ones_like(XX) * dx + np.ones_like(YY) * dy)

    XYZ0 = np.stack((XX, YY, ZZ0), axis=-1).reshape(-1, 3)
    XYZ1 = np.stack((XX, YY, ZZ1), axis=-1).reshape(-1, 3)

    XYZ = np.stack((XYZ0, XYZ1)).reshape(-1,3)

    XYZ = swapXY(XYZ)

    return XYZ

def vec2mat(w):
    theta = np.linalg.norm(w)

    k = w / theta

    kx = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta)*kx + (1 - np.cos(theta))*kx@kx

    return R

def poseVector2TransformationMatrix(pose):
    w = pose[:3]
    t = pose[3:]
    R = vec2mat(w) #R = np.array(Rotation.from_rotvec(w).as_matrix())
    Rt = np.concatenate([R, t.reshape(1,3).T], axis=1)
    Rt = np.concatenate([Rt, np.array([[0, 0, 0, 1]])], axis=0)
    return Rt

def projectPoints(K, Rt, Pw):
    Pw = np.concatenate([Pw, np.ones((1,len(Pw))).T], axis=1)

    p = K @ (Rt @ Pw.T)[:-1]

    p = p / p[2]

    return p[:2]

def distort(p, K, D):
    pc = K[0:2,2].reshape(2,1)

    pd = np.zeros((2, len(p[0])))

    r2 = (p[0] - pc[0])**2 + (p[1] - pc[1])**2
    r4 = r2**2

    pd = (1 + D[0]*r2 + D[1]*r4) * (p - pc) + pc

    return pd

def load_data():
    K = open("./data/K.txt").read()
    D = open("./data/D.txt").read()
    poses = open("./data/poses.txt").read()
    filenames = glob.glob('./data/images/*.jpg')

    K = txt2array(K).reshape(3,3)
    D = txt2array(D)
    P = txt2array(poses).reshape(len(poses.split('\n')[:-1]), 6)

    return K, D, P, filenames

def main():
    K, D, P, filenames = load_data()

    img = './data/images_undistorted/img_0001.jpg'
    img = cv2.imread(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Rt = poseVector2TransformationMatrix(P[0])

    corners_w = create_corner_dots()
    corners_px = projectPoints(K, Rt, corners_w)
    img = add_corners(img, corners_px)

    cube_w = create_cube_dots(xy=(2,3), size=3)
    cube_px = projectPoints(K, Rt, cube_w)
    img = add_cube(img, cube_px)

    imshow(img)

    img = cv2.imread(filenames[0])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners_px = distort(corners_px, K, D)
    #print(corners_px)
    img = add_corners(img, corners_px)
    #print(img.shape)
    imshow(img)

    plt.imshow(img)
    plt.show()

if __name__=="__main__":
    main()
