import os 
import copy 
import numpy as np 
from scipy.linalg import expm, logm, sinm, cosm
from scipy.optimize import leastsq
from matplotlib import pyplot as plt 

#from .linalgs import Homogenous, Homo
#from .hello import hello 

DIR_DATA = '../data'

def loadtxt(filename):
    if '.txt' not in filename: filename += '.txt' 
    return np.loadtxt(os.path.join(DIR_DATA, filename))

def npprint(array, name='array'):
    print()
    print(f'{name}{array.shape, array.dtype} = ')
    print(array)  

def title(name='title', length=100, marker='*'):
    print()
    print(length*marker) 
    print(f'\t {name.upper()}') 
    print(length*marker) 

def subtitle(name='subtitle', length=100, marker='-'):
    print() 
    print(length*marker) 
    print(f'\t {name}') 
    print(length*marker) 

def comment(txt='comment'):
    print('"' + txt + '"')

def Homogenous(points):
    ones = np.ones(points.shape[1]) 
    return np.vstack((points, ones)) 

def HomogMatrix2twist(H):
    # careful for rotations of pi; the top 3x3 submatrix of the returned
    # se_matrix by logm is not skew-symmetric (bad).
    se_matrix = logm(H)
    v = se_matrix[:3,3]
    w = Matrix2Cross(se_matrix[:3,:3])
    twist = np.hstack((v, w))
    return twist 

def twist2HomogMatrix(x):
    v = x[:3]; w = x[3:] 
    M = np.hstack((Cross2Matrix(w),v.reshape(3,1)))
    se_matrix = np.vstack((M,np.array([0,0,0,0])))
    H = expm(se_matrix)
    return H

def Matrix2Cross(M):
    return np.array([-M[1,2],M[0,2],-M[0,1]]) 

def Cross2Matrix(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]) 

def cropProblem(hidden_state, observations, ground_truth, cropped_num_frames):
    num_frames = observations[0].astype(int)

    obs_id = 2 
    for i in range(cropped_num_frames):
        num_kpts = observations[obs_id].astype(int) 

        if i+1 == cropped_num_frames:
            id_kpts = observations[obs_id+1+2*num_kpts:obs_id+1+3*num_kpts].astype(int)
            cropped_num_lmks = np.max(id_kpts)
           
        obs_id += 1 + 3*num_kpts
    
    
    cropped_twists = hidden_state[:6*cropped_num_frames] 
    cropped_P_lmks = hidden_state[6*num_frames:6*num_frames+3*cropped_num_lmks]
    cropped_hidden_state = np.hstack((cropped_twists, cropped_P_lmks)) 

    cropped_nums = np.array([cropped_num_frames, cropped_num_lmks])
    cropped_Os = observations[2:obs_id]
    cropped_observations = np.hstack((cropped_nums, cropped_Os)) 

    cropped_ground_truth = ground_truth[:,:cropped_num_frames] 

    return cropped_hidden_state, cropped_observations, cropped_ground_truth 

def alignEstimateToGroundTruth(pp_G_C, p_V_C):
    # Returns the points of the estimated trajectory p_V_C transformed into the
    # ground truth frame G. The similarity transform Sim_G_V is to be chosen
    # such that it results in the lowest error between the aligned trajectory
    # points p_G_C and the points of the ground truth trajectory pp_G_C. All
    # matrices are 3xN.

    # make an initial guess 
    twist_guess = HomogMatrix2twist(np.eye(4))
    scale_guess = np.array(1) 
    x0 = np.hstack((twist_guess, scale_guess)) 
    #>> x0((7,), dtype('float64')) = [ 0.  0.  0. -0.  0. -0.  1.]

    # non-linear least squares 
    err = lambda x: alignmentError(x, pp_G_C, p_V_C) 
    x, flag = leastsq(err, x0)

    # return the estimated trajectory 
    p_G_C = model(x, p_V_C)  

    return p_G_C

def model(x, p_V_C):
    twist_guess = x[:6] 
    scale_guess = x[6] 
    matrix_guess = twist2HomogMatrix(twist_guess) 

    R = matrix_guess[:3,:3]
    t = matrix_guess[:3,3]
    sR = scale_guess * R 

    M = np.hstack((sR, t.reshape(3,1)))
    M = np.vstack((M, np.array([0,0,0,1]).reshape(1,-1)))
    p_G_C = M @ Homogenous(p_V_C) 
    p_G_C = p_G_C[:-1] 

    return p_G_C

def alignmentError(x, pp_G_C, p_V_C):
    p_G_C = model(x, p_V_C)  

    errors = pp_G_C - p_G_C
 
    #error = np.sqrt((errors**2).sum(axis=0)).T
    error = errors.flatten()

    return error

def projection(K, p_C_L):
    p = K @ p_C_L 
    return p[:-1]/p[-1] 

def baError(hidden_state, observations, K): # bundle adjustment 
    num_frames, num_lmks = observations[:2].astype(int) 
    twists = hidden_state[:6*num_frames].reshape(-1,6) 
    p_W_lmks = hidden_state[-3*num_lmks:].reshape(-1,3).T 
    #print(num_frames, 'num_frames') 
    #print(num_lmks, 'num_lmks')

    err_terms = [] 
    obs_id = 2 
    for i in range(num_frames):
        num_kpts = observations[obs_id].astype(int) 
        p_kpts = observations[obs_id+1:obs_id+1+2*num_kpts].reshape(-1,2).T
        p_kpts = np.flipud(p_kpts)
        id_kpts = observations[obs_id+1+2*num_kpts:obs_id+1+3*num_kpts].astype(int) - 1
        obs_id += 1 + 3*num_kpts

        p_W_L = p_W_lmks[:,id_kpts]  
        num_lmks = p_W_L.shape[1] 
        T_W_C = twist2HomogMatrix(twists[i]) 
        #p_C_L = T_W_C[:3,:3].T @ p_W_L - T_W_C[:3,3].reshape(3,1)

        # T_C_W = T_W_C.T 
        # p_C_L = T_C_W[:3,:3] @ p_W_L + T_C_W[:3,3].reshape()
        p_C_L = (T_W_C.T @ Homogenous(p_W_L))[:-1]

        #npprint(p_C_L)
        #print(num_kpts, 'num_kpts') 
        #npprint(p_C_L, 'p_C_L') 
        #npprint(p_W_L, 'p_W_L') 

        projections = projection(K, p_C_L)
        error_term = p_kpts - projections
        err_terms.extend(error_term.flatten().tolist())  

    return err_terms 

def runBA(hidden_state, observations, K):        
    #options = optimoptions(@lsqnonlin, 'Display', 'iter', 'MaxIter', 20)
    # 'iter' — 各反復の出力を表示し、既定の終了メッセージを与える。
    # MaxIterations: 可能な反復の最大数 (正の整数)。既定値 400 です。
    err = lambda x: baError(x, observations, K) 
    hidden_state = leastsq(err, hidden_state, maxfev=20)  

    return hidden_state 

def runBAadvanced(hidden_state, observations, K):
    with_pattern = False 
    if with_pattern:
        num_frames = observations[0].astype(int) # 250
        num_obs = (observations.shape[0] - 2 - num_frames) // 3 #125634
        '''
        num_frames:  250
        num_obs:  125634
        observations.shape[0] - 2 - num_frames: 376902
        '''
        num_err_terms = 2 * num_obs 

        for i in range(num_frames):
            num_kpts = observations[0].astype(int) 
        
    #options = optimoptions(@lsqnonlin, 'Display', 'iter', 'MaxIter', 20)
    # 'iter' — 各反復の出力を表示し、既定の終了メッセージを与える。
    # MaxIterations: 可能な反復の最大数 (正の整数)。既定値 400 です。
    err = lambda x: baError(x, observations, K) 
    hidden_state = leastsq(err, hidden_state, maxfev=20)  

    return hidden_state 

def spalloc():
    return 0 
