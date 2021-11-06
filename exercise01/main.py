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

def imshow2(imgs):
    interp = ["without bilinear interpolation", "with bilinear interpolation"]
    fig, axs = plt.subplots(1, len(imgs), figsize=(20, 5)) 
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        ax.set_title(interp[i])
    plt.show()

def imshow2_vertical(imgs):
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(10, 10)) 
    axs[0].set_title('without bilinear interpolation')
    axs[0].imshow(imgs[0])

    axs[1].set_title('with bilinear interpolation')
    axs[1].imshow(imgs[1])
    plt.show()

def add_corners(img, p, r=5, color=(0,0,255), thickness=-1):
    for x,y in zip(p[0], p[1]):
        img = cv2.circle(img, (round(x), round(y)), r, color, thickness)
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

    YXZ = swapXY(XYZ) 

    return YXZ 

def create_cube_dots(xy=(0,0), size=1, dx=0.04, dy=0.04):

    X = np.array([xy[0], xy[0]+size]) * dx 
    Y = np.array([xy[1], xy[1]+size]) * dy

    XX, YY = np.meshgrid(X, Y) 
    ZZ0 = XX*0 + YY*0  
    ZZ1 = -0.5 * size * (np.ones_like(XX) * dx + np.ones_like(YY) * dy)

    XYZ0 = np.stack((XX, YY, ZZ0), axis=-1).reshape(-1, 3)
    XYZ1 = np.stack((XX, YY, ZZ1), axis=-1).reshape(-1, 3)

    XYZ = np.stack((XYZ0, XYZ1)).reshape(-1,3)

    YXZ = swapXY(XYZ)

    return YXZ

def vec2mat(w):
    theta = np.linalg.norm(w)

    k = w / theta

    kx = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx@kx)
    
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

    r2 = (p[0] - pc[0])**2 + (p[1] - pc[1])**2
    r4 = r2**2

    pd = (1 + D[0]*r2 + D[1]*r4) * (p - pc) + pc 

    return pd 

def undistort(img_d, K, D, bilinear_interpolation=False): 
    img = np.zeros_like(img_d) 
    (height, width, _) = img.shape 

    for v in range(height): 
        for u in range(width):
            [ud, vd] = distort(np.array([[u], [v]]), K, D).reshape(2) 
            [u1, v1] = np.array(list(map(int, np.floor([ud, vd]))))

            if bilinear_interpolation:
                if 0 <= u1 and u1+1 < width and 0 <= v1 and v1+1 < height:
                    [a, b] = [ud - u1, vd -v1] 
                    img[v, u] = (1-b)*((1-a)*img_d[v1, u1] + a*img_d[v1, u1+1]) + b * ((1-a)*img_d[v1+1, u1] + a*img_d[v1+1, u1+1])

            else:
                if 0 <= u1 and u1 < width and 0 <= v1 and v1 < height:
                    img[v, u] = img_d[v1, u1] 
    return img 

def load_data():
    K = open("./data/K.txt").read()
    D = open("./data/D.txt").read() 
    poses = open("./data/poses.txt").read()
    filenames = glob.glob('./data/images/*.jpg') 

    K = txt2array(K).reshape(3,3) 
    D = txt2array(D) 
    P = txt2array(poses).reshape(len(poses.split('\n')[:-1]), 6)
    filenames = sorted(filenames) 

    return K, D, P, filenames


def show_undistorted_images(img, K, D):
    img0 = undistort(img, K, D, bilinear_interpolation=False)  
    imshow(img0, "undistorted") 

    img1 = undistort(img, K, D, bilinear_interpolation=True)  
    imshow(img1, "undistorted") 

    imshow2([img0, img1]) 

def show_movie(P, K, D, filenames):
    for i in range(len(filenames)):
        img = cv2.imread(filenames[i]) 
        Rt = poseVector2TransformationMatrix(P[i])

        corners_w = create_corner_dots()
        corners_px = projectPoints(K, Rt, corners_w)
        img = add_corners(img, distort(corners_px, K, D))

        cube_w = create_cube_dots(xy=(2,3), size=3)
        cube_px = projectPoints(K, Rt, cube_w) 
        img = add_cube(img, distort(cube_px, K, D)) 

        imshow(img, "distorted") 

def create_movie(P, K, D, filenames, shape=(480, 720, 3), fps=30, videoname='video.avi'):
    height, width, layers = shape 
    size = (height, width)
    frame_array = np.zeros((len(filenames), height, width, layers))

    for i in range(len(filenames)):
        img = cv2.imread(filenames[i]) 
        Rt = poseVector2TransformationMatrix(P[i])

        corners_w = create_corner_dots()
        corners_px = projectPoints(K, Rt, corners_w)
        img = add_corners(img, distort(corners_px, K, D))

        cube_w = create_cube_dots(xy=(2,3), size=3)
        cube_px = projectPoints(K, Rt, cube_w) 
        img = add_cube(img, distort(cube_px, K, D)) 
        
        frame_array[i] = img

    out = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

def main():
    K, D, P, filenames = load_data()  

    ### undistorted image ###

    img_undistorted = cv2.imread('./data/images_undistorted/img_0001.jpg')  
    Rt = poseVector2TransformationMatrix(P[0])

    corners_w = create_corner_dots()
    corners_px = projectPoints(K, Rt, corners_w)
    img_undistorted = add_corners(img_undistorted, corners_px) 

    cube_w = create_cube_dots(xy=(2,3), size=3)
    cube_px = projectPoints(K, Rt, cube_w) 
    img_undistorted = add_cube(img_undistorted, cube_px) 
    
    imshow(img_undistorted, "undistorted") 
    plt.imshow(img_undistorted); plt.show() 

    ### distorted images ### 

    img = cv2.imread(filenames[0]) 
    img = add_corners(img, distort(corners_px, K, D))
    img = add_cube(img, distort(cube_px, K, D)) 

    imshow(img, "distorted")  
    plt.imshow(img); plt.show() 

    ### undistortion ### 
    show_undistorted_images(img, K, D) 
    
    ### movie ###
    show_movie(P, K, D, filenames)
    create_movie(P, K, D, filenames, shape=img.shape) 

if __name__=="__main__":
    main() 