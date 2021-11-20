import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import glob

def main():
    # parameters 
    args = {
        'K': '../data/K.txt',
        'D': '../data/D.txt',
        'poses': '../data/poses.txt',
        'images': '../data/images/*.jpg',
        'undistorted_image': '../data/images_undistorted/img_0001.jpg',
        'grid_shape': (6,9),
        'unit_size': 0.04,
    }

    # Load data 
    K = txt2array(args['K']).reshape(3,3)           # camera parameters 
    D = txt2array(args['D'])                        # distortion parameters 
    poses = txt2array(args['poses']).reshape(-1, 6) # camera poses 
    filenames = sorted(glob.glob(args['images']))   # image file names 

    # Show loaded data 
    if 1:
        print('\n K=\n', K)
        print('\n D=\n', D)
        print('\n poses=\n', poses)
        print('\n image names=\n') 
        for i in range(len(filenames)): print(filenames[i]) 
    
    if 0:
        # undistorted image    
        img_undistorted = imread(args['undistorted_image']) 
        imshow(img_undistorted)
        imsave(img_undistorted, '../results/undistorted.png')
    
        # add grid corners 
        Rt = poseVector2TransformationMatrix(poses[0])
        cornersXYZ = createGridCorners(args['grid_shape'], args['unit_size']) 
        cornersxy = projectPoints(K, Rt, cornersXYZ)

        img_undistorted = imdots(img_undistorted, cornersxy) 
        imshow(img_undistorted)
    
        # add a cube 
        cubeXYZ, corr = createCubeCorners(Oyx=(2,3), size=3, unit_size=args['unit_size']) 
        cubexy = projectPoints(K, Rt, cubeXYZ) 
        img_undistorted = imcube(img_undistorted, cubexy, corr) 
        imshow(img_undistorted, "undistorted") 
        imsave(img_undistorted, '../results/cube_dots_undistorted.png')
        #plt.imshow(img_undistorted); plt.show() 
    
    if 0:
        ### distorted images ### 
        Rt = poseVector2TransformationMatrix(poses[0])
        cornersXYZ = createGridCorners(args['grid_shape'], args['unit_size']) 
        cornersxy = projectPoints(K, Rt, cornersXYZ)
        cubeXYZ, corr = createCubeCorners(Oyx=(2,3), size=3, unit_size=args['unit_size']) 
        cubexy = projectPoints(K, Rt, cubeXYZ) 

        img = imread(filenames[0]) 
        img = imdots(img, distort(cornersxy, K, D))
        img = imcube(img, distort(cubexy, K, D),  corr) 

        imshow(img, "distorted")  
        imsave(img_undistorted, '../results/cube_dots_distorted.png')
        plt.imshow(img); plt.show() 
    
    if 0:
        ### undistortion ### 
        img = imread(filenames[0]) 
        img0 = undistort(img, K, D, bilinear_interpolation=False)  
        imshow(img0, "undistorted") 

        img1 = undistort(img, K, D, bilinear_interpolation=True)  
        imshow(img1, "undistorted") 

        imcompare(img0, img1, 'without bilinear interpolation', 'with bilinear interpolation')   
    
    if 0:
        ### movie ###
        show_movie(poses, K, D, filenames)
        create_movie(poses, K, D, filenames, shape=img.shape) 

def distort(p, K, D):
    '''
    Input:
        - p(2, N): pixel coordinate of input points 
        - K(3,3): the camera parameters 
        - D(2)
    
    Output:
        - pd(2, N): distorted pixel coordinates of points 
    '''
    pc = K[0:2,-1].reshape(2,1)                     # the center of distortion from K 

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


def createGridCorners(shape=(6,9), unit_size=0.04):

    Y = np.arange(shape[0]) * unit_size 
    X = np.arange(shape[1]) * unit_size

    YY, XX = np.meshgrid(Y, X) 
    ZZ = YY*0 + XX*0  
 
    XYZ = np.stack((XX, YY, ZZ), axis=-1).reshape(-1, 3).T

    if 1:
        print('XX = \n', XX) 
        print('YY = \n', YY) 
        print('ZZ = \n', ZZ)
        print('np.stack((XX,YY,ZZ),axis=-1)=\n', np.stack((XX, YY, ZZ), axis=-1))
        print('XYZ = \n', XYZ) 
        
    return XYZ 

def createCubeCorners(Oyx=(0,0), size=1, unit_size=0.04):

    Y = np.array([Oyx[0], Oyx[0]+size]) * unit_size # vertical 
    X = np.array([Oyx[1], Oyx[1]+size]) * unit_size # horizontal 

    YY, XX = np.meshgrid(Y, X) 
    ZZbot = YY*0 + XX*0  
    ZZtop = -0.5 * size * (np.ones_like(YY) * unit_size + np.ones_like(XX) * unit_size) 

    XYZbot = np.stack((XX, YY, ZZbot), axis=-1).reshape(-1, 3).T 
    XYZtop = np.stack((XX, YY, ZZtop), axis=-1).reshape(-1, 3).T 

    XYZ = np.concatenate((XYZbot, XYZtop),axis=-1)

    corr = np.array([[0, 0, 1, 2, 4, 4, 5, 6, 0, 1, 2, 3],
                    [1, 2, 3, 3, 5, 6, 7, 7, 4, 5, 6, 7]],dtype=np.uint8)  

    return XYZ, corr 

def projectPoints(K, Rt, Pw):
    '''project points onto screen 
    Input:
        - K(3,3): a camera matrix
        - Rt(3,4): a transformatino matrix 
        - Pw(3, N): N 3D world coodinates of points 
    
    Output:
        - p(2, N): N pixel coordinates of points 
    '''
    
    p = K @ Rt @ homogenous(Pw)
     
    p /= p[2]
    
    return p[:2]

def poseVector2TransformationMatrix(pose):

    R = rotVec2Mat(pose[:3])  
    t = pose[3:].reshape(3,1) 
    Rt = np.hstack((R, t))

    if 0:
        print(R) 
        print(t)  
        print(Rt) 
    
    return Rt

def homogenous(X):
    '''
    Input:
        - X(n, N): N n-dimensional coordinates 

    Output:
        - Xh(n+1, N): their hogenous coordinates 
    '''
    return np.vstack((X, np.ones(X.shape[1]))) 

def rotVec2Mat(w): 
    theta = np.linalg.norm(w)

    k = w / theta

    kx = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx@kx)
    
    return R 

def imdots(img, p, r=5, color=(0,0,255), thickness=-1):
    for x,y in zip(p[0], p[1]):
        img = cv2.circle(img, (round(x), round(y)), r, color, thickness)
    return img 

def imcube(img, p, corr, r=5, color=(0,0,255)):
    for i, j in zip(corr[0], corr[1]):
        x0 = round(p[0, i]); y0 = round(p[1, i])
        x1 = round(p[0, j]); y1 = round(p[1, j])
        img = cv2.line(img, (x0, y0), (x1, y1), color, r)
    
    return img 

def implotXYZ(XX, YY, ZZ): # ZZ = f(XX, YY) 
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(XX, YY, ZZ)
    plt.show()

def imcomparev(img1, img2, name1='img1', name2='img2', figsize=(10,10)): 
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=figsize)  

    axs[0].set_title(name1)
    axs[0].imshow(img1)

    axs[1].set_title(name2)
    axs[1].imshow(img2)
    plt.show()


def imcompare(img1, img2, name1='img1', name2='img2', figsize=(20,5)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)  
    
    axs[0].imshow(img1)
    axs[0].set_title(name1)

    axs[1].imshow(img2)
    axs[1].set_title(name2)
 
    plt.show()

    #plt.savefig(title) 

def imshow(img, name="img"):
    cv2.imshow(name, img)
    while True:
        if cv2.waitKey(1)==ord('q'): break 
    cv2.destroyAllWindows()

def imsave(img, name='img.png'):
    cv2.imwrite(name, img) 

def imread(filename):
    return cv2.imread(filename)  

def txt2array(filename, sep=' '):
    txt = open(filename).read() 
    return np.fromstring(txt, dtype=float, sep=sep)


def show_movie(P, K, D, filenames):
    for i in range(len(filenames)):
        img = cv2.imread(filenames[i]) 
        Rt = poseVector2TransformationMatrix(P[i])

        corners_w = createGridCorners() 
        corners_px = projectPoints(K, Rt, corners_w)
        img = imdots(img, distort(corners_px, K, D))

        cube_w = createCubeCorners(xy=(2,3), size=3)
        cube_px = projectPoints(K, Rt, cube_w) 
        img = imcube(img, distort(cube_px, K, D)) 

        imshow(img, "distorted") 

def create_movie(P, K, D, filenames, shape=(480, 720, 3), fps=30, videoname='video.avi'):
    height, width, layers = shape 
    size = (height, width)
    frame_array = np.zeros((len(filenames), height, width, layers))

    for i in range(len(filenames)):
        img = cv2.imread(filenames[i]) 
        Rt = poseVector2TransformationMatrix(P[i])

        corners_w = createGridCorners()
        corners_px = projectPoints(K, Rt, corners_w)
        img = imdots(img, distort(corners_px, K, D))

        cube_w = createCubeCorners(xy=(2,3), size=3)
        cube_px, corr = projectPoints(K, Rt, cube_w) 
        img = imcube(img, distort(cube_px, K, D)) 
        
        frame_array[i] = img

    out = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__=="__main__":
    main() 
