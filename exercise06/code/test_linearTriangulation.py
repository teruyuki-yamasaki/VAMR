import numpy as np 
from linearTriangulation import linearTriangulation 

def run_test_linearTriangulatin(func):
    # Test linear triangulation
    
    N = 10 # Number of 3-D points

    P = np.random.randn(4, N)
    P[2, :] = P[2, :] * 5 + 10 
    P[3, :] = 1     

    M1 =  np.array([
            [500, 0, 320, 0],
            [0, 500, 240, 0],
            [0, 0, 1, 0]])

    M2 =  np.array([
            [500, 0, 320, -100,],
            [0, 500, 240, 0],
            [0, 0, 1, 0]]) 
                    
    p1 = M1 @ P    #% Image (i.e., projected) points
    p2 = M2 @ P

    P_est = func(p1,p2,M1,M2)

    print('P_est-P=\n')
    print(P_est-P.T)

if __name__=="__main__":
    run_test_linearTriangulatin(linearTriangulation)
