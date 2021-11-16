def visualize(img1, img2, p1, p2, P, R_C2_W, T_C2_W): 
    # Visualize the 3-D scene
    plt.figure(1)
    plt.subplot(1,3,1)

    # R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

    # P is a [4xN] matrix containing the triangulated point cloud (in homogeneous coordinates), 
    # given by the function linearTriangulation
    plot3(P[0,:], P[1,:], P[2,:], 'o');

    # Display camera pose

    plotCoordinateFrame(np.eye(3), np.zeros((3,1)), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

    center_cam2_W = -R_C2_W.T @ T_C2_W;
    plotCoordinateFrame(R_C2_W.T, center_cam2_W, 0.8);
    text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

    plt.axis equal
    rotate3d on;
    plt.grid

    # Display matched points
    immatches(img1, img2, p1, p2) 


def run_sfm(estimateEssentialMatrix, decomposeEssentialMatrix, disambiguateRelativePose):

    img1 = imread('./data/0001.jpg')
    img2 = imread('./data/0002.jpg')
    
    K = np.array([
            [1379.74, 0, 760.35],
            [0, 1382.08, 503.41],
            [0, 0, 1]]) 
    
    # Load outlier-free point correspondences

    p1 = txt2array('./data/matches0001.txt').reshape(2,-1)
    p2 = txt2array('./data/matches0002.txt').reshape(2,-1)

    p1 = homogenous(p1)
    p2 = homogenous(p2) 

    # Estimate the essential matrix E using the 8-point algorithm

    E = estimateEssentialMatrix(p1, p2, K, K);

    # Extract the relative camera positions (R,T) from the essential matrix

    # Obtain extrinsic parameters (R,t) from E
    [Rots, u3] = decomposeEssentialMatrix(E);

    # Disambiguate among the four possible configurations
    [R_C2_W, T_C2_W] = disambiguateRelativePose(Rots, u3, p1, p2, K, K);

    # Triangulate a point cloud using the final transformation (R,T)
    M1 = K @ np.eye(3,4);
    M2 = K @ transformationMat(R_C2_W, T_C2_W)
    P = linearTriangulation(p1, p2, M1, M2)

    visualize(img1, img2, p1, p2, P, R_C2_W, T_C2_W)
