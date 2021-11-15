import numpy as np 
from test_8point import run_test_8point

def fundamentalEightPoint(p1,p2):
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

    F = np.eye(3) #np.zeros((3,3))

    return F

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

    F = np.eye(3) 

    return F 

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

    i = 5
    p1 = p1[:,:i]
    p2 = p2[:,:i]
    print('\n p1=')
    print(p1)
    print('\n p2=')
    print(p2)

    #homog_points = np.hstack([p1, p2]).reshape(3,2,-1);
    homog_points = np.hstack([p1, p2]) #.reshape(3,2,-1);
    print('\n homog_points=')
    print(homog_points*1e-3)

    #epi_lines = [F.'*p2, F*p1];
    epi_lines = np.hstack([F.T @ p2, F @ p1]);
    print('\n epi_lines=')
    print(epi_lines*1e-3)

    #denom = epi_lines(1,:).^2 + epi_lines(2,:).^2;
    #cost = sqrt( sum( (sum(epi_lines.*homog_points,1).^2)./denom ) / NumPoints );

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2;
    #print(epi_lines.shape)
    #print(homog_points.shape)
    #print((epi_lines * homog_points).shape)
    a = np.sum(epi_lines * homog_points, axis=0)
    print('\n a=')
    print(a)
    cost = np.sqrt( np.sum(a**2 /denom ) / NumPoints );
    print('\n cost=')
    print(cost) 
    return cost 

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

    E = np.zeros((3,3)) 

    return E 

if __name__=="__main__":
    run_test_8point(fundamentalEightPoint, distPoint2EpipolarLine)
