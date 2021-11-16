import numpy as np 

def homogenous(p): 
    # returns homogenous coordinates of n-D points
    # given coordinates of n-D points  

    # Input:
    #   - p(n, N): coordinates of n-D points
    # 
    # Output:
    #   - ph(n+1, N): homogenous coordinates of n-D points 

    return np.vstack((p, np.ones(p.shape[1]))) 

def vec(F):
    # stack vec F's columns into a single column vector 

    # Input:
    #   - F(3,3): a 3x3 matrix 
    # 
    # Output:
    #   - vecF(9): a 9D vector 

    return F.T.flatten()

def unvec(F):
    # do the oposite operation of vec(F) 
    # decompose vec F into three columns 

    # Input:
    #   - F(9): a 9-D vector 
    # 
    # Output:
    #   - matF(3,3): a 3x3 matrix  

    return F.reshape(3,3).T 

def krons(p1, p2):
    # stack the Kronecker products of N 2-D homogenous coodinates into a matrix 

    # Input:
    #   - p1(3, N): 
    #   - p2(3, N):
    # 
    # Output:
    #   - Q(N, 9):

    N = p1.shape[1]
    Q = np.zeros((N, 9)) # dtype ?? 

    for i in range(N):
        Q[i] = np.kron(p1[:,i], p2[:,i]) 
    
    return Q 
