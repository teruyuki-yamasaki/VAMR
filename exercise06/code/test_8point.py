import numpy as np 
from linalgs import *

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
        print('\n homog_points=\n', homog_points*1e-3)
        print('\n epi_lines=\n', epi_lines*1e-3)
        print('\n a=\n', a)
        print('\n cost=\n', cost) 
        print(denom.shape) # (80,)
        print(a.shape) # (80,)
        print(b.shape) # (3, 80)

    return cost 

def run_test_8point(fundamentalEightPoint, fundamentalEightPoint_normalized, distPoint2EpipolarLine):
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

    sigma = 1e-1;
    noisy_x1 = x1 + sigma * np.random.randn(x1.shape[0]).reshape(-1,1);
    noisy_x2 = x2 + sigma * np.random.randn(x2.shape[0]).reshape(-1,1);

    # Estimate fundamental matrix
    # Call the 8-point algorithm on noisy inputs x1,x2
    F = fundamentalEightPoint(noisy_x1,noisy_x2);

    # Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
    cost_algebraic = sumSquaredEpipolarConstraints(F, noisy_x1, noisy_x2) 
    cost_dist_epi_line = distPoint2EpipolarLine(F, noisy_x1, noisy_x2);

    print('Noisy correspondences (sigma=%f), with fundamentalEightPoint\n' % sigma);
    print('Algebraic error: %f \n' % cost_algebraic);
    print('Geometric error: %f px\n\n' % cost_dist_epi_line);

    ## Normalized 8-point algorithm
    # Call the normalized 8-point algorithm on inputs x1,x2
    Fn = fundamentalEightPoint_normalized(noisy_x1,noisy_x2);

    # Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
    cost_algebraic = sumSquaredEpipolarConstraints(Fn, noisy_x1, noisy_x2)
    cost_dist_epi_line = distPoint2EpipolarLine(Fn, noisy_x1, noisy_x2)

    print('Noisy correspondences (sigma=%f), with fundamentalEightPoint_normalized\n' % sigma);
    print('Algebraic error: %f \n' % cost_algebraic);
    print('Geometric error: %f px\n\n' % cost_dist_epi_line)

from 8point import fundamentalEightPoint, fundamentalEightPoint_normalized, distPoint2EpipolarLine
if __name__=='__main__':
    run_test_8point(fundamentalEightPoint, fundamentalEightPoint_normalized, distPoint2EpipolarLine)
