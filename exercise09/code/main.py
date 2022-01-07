import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq 
from scipy.linalg import expm, logm, sinm, cosm 
import copy 

import algs 

def main():
    # ---------------------------------------------------
    # PART 0: UNDERSTAND INPUT DATA 
    # ---------------------------------------------------
    algs.title('part 0: understand input data') 

    # raw input 
    K = algs.loadtxt('K') 
    poses = algs.loadtxt('poses')
    observations = algs.loadtxt('observations') 
    hidden_state = algs.loadtxt('hidden_state')
    pp_G_C = poses[:,[3,7,11]].T # Ground Truth 

    # ---------------------------------------------------
    # show raw data
    # ---------------------------------------------------
    algs.subtitle('raw data')
    algs.npprint(K, 'K')
    algs.npprint(poses, 'poses')
    algs.npprint(observations, 'observations')
    algs.npprint(hidden_state, 'hidden_state')  
    algs.npprint(pp_G_C[:,:4], 'pp_G_C (top4)')
    algs.comment('Ground Truth: poses[:,[3,7,11]].T')

    orgs = {
        'K': K,
        'poses': poses,
        'observations': observations,
        'hidden_state': hidden_state,
        'pp_G_C': pp_G_C
    }

    # ---------------------------------------------------
    # extract data 
    # ---------------------------------------------------
    algs.subtitle('extracted data')    

    # ---------------------------------------------------
    # before cropping 
    # ---------------------------------------------------
    num_frames, num_lmks = observations[:2].astype(int) 
    T_W_C = hidden_state[:6*num_frames].reshape(-1,6).T # the camera pose of each frame in twist 
    P_W_lmks = hidden_state[6*num_frames:].reshape(-1,3).T # the 3D positions of landmarks 

    print('\n the number of frames = ', num_frames) 
    print('\n the number of landmarks = ', num_lmks)
    algs.npprint(pp_G_C[:,:4], 'pp_G_C (top4)')
    algs.comment('Ground Truth: poses[:,[3,7,11]].T')
    algs.npprint(T_W_C[:,:4], 'T_W_C (top4)')
    algs.comment('twists: hidden_state[:6*num_frames].reshape(-1,6).T')
    algs.npprint(P_W_lmks, 'P_W_lmks')
    algs.comment('3D positions of lmks: hidden_state[-3*num_lmks:].reshape(-1,3).T')

    # ---------------------------------------------------
    # cropping 
    # ---------------------------------------------------
    cropped_num_frames = 150
    hidden_state, observations, pp_G_C = algs.cropProblem(
        orgs['hidden_state'], orgs['observations'], orgs['pp_G_C'], 
        cropped_num_frames)
    algs.comment(f'problem cropped with cropped_num_frames = {cropped_num_frames}')

    num_frames, num_lmks = observations[:2].astype(int) 
    T_W_C = hidden_state[:6*num_frames].reshape(-1,6).T # the camera pose of each frame in twist 
    P_W_lmks = hidden_state[6*num_frames:].reshape(-1,3).T # the 3D positions of landmarks 

    print('\n the number of frames = ', num_frames) 
    print('\n the number of landmarks = ', num_lmks)
    algs.npprint(pp_G_C[:,:4], 'pp_G_C (top4)')
    algs.comment('Ground Truth: poses[:,[3,7,11]].T')
    algs.npprint(T_W_C[:,:4], 'T_W_C (top4)')
    algs.comment('twists: hidden_state[:6*num_frames].reshape(-1,6).T')
    algs.npprint(P_W_lmks, 'P_W_lmks')
    algs.comment('3D positions of lmks: hidden_state[-3*num_lmks:].reshape(-1,3).T')
    
    # ---------------------------------------------------
    # show data in each frame 
    # ---------------------------------------------------
    algs.subtitle('info in each frame') 
    n = 3
    obs_id = 2
    for i in range(n):
        num_kpts = observations[obs_id].astype(int) 
        p_kpts = np.flipud(observations[obs_id+1:obs_id+1+2*num_kpts].reshape(-1,2).T) 
        id_kpts = observations[obs_id+1+2*num_kpts:obs_id+1+3*num_kpts].astype(int) 
        obs_id += 1 + 3*num_kpts 

        if 1:
            print(str('-'*30)+ str(f'frame{i}') + str('-'*30))
            print(f'the number of kpts observed in frame {i} = ', num_kpts) 
            algs.npprint(id_kpts, 'the indices of the observed lmks: ')
            algs.comment('observations[obs_id].astype(int)')
            algs.npprint(p_kpts[:,:4], 'the 2D positions of the lmks (top4): ') 
            algs.comment('np.flipud(observations[obs_id+1:obs_id+1+2*num_lmks_observed].reshape(-1,2).T)')
            
        if 0: 
            plt.scatter(p_kpts[0], p_kpts[1], marker='.')  
            plt.axis('equal') 
            plt.title(f'observed kpts in frame {i}')
            plt.show()

        tau = T_W_C[:,i]
        v = tau[:3] # linear part 
        w = tau[3:] # angular part 
        R = algs.Cross2Matrix(w) 
        M0 = np.vstack((np.hstack((R,v.reshape(3,-1))),np.array([0,0,0,1])))
        M = algs.twist2HomogMatrix(tau) 

        if 1:
            algs.npprint(tau, 'tau')
            algs.comment('twist: T_W_C[:,i]') 
            algs.npprint(v, 'v')
            algs.comment('linear part: tau[:3]') 
            algs.npprint(w, 'w')
            algs.comment('angular part: tau[3:]')
            algs.npprint(R, 'R')
            algs.comment('algs.Cross2Matrix(w)')
            algs.npprint(M0, 'M0')
            algs.comment('np.vstack((np.hstack((R,v.reshape(3,-1))),np.array([0,0,0,1])))')
            algs.npprint(M, 'M')
            algs.comment('algs.twist2HomogMatrix(tau)') 

    # ---------------------------------------------------
    # Compare estimated trajectory to ground truth
    # ---------------------------------------------------
    algs.subtitle('Compare estimated trajectory to ground truth')  
    p_V_C = np.zeros((3,num_frames))
    for i in range(num_frames):
        each_T_V_C = algs.twist2HomogMatrix(T_W_C[:,i])
        p_V_C[:,i] = each_T_V_C[:3,-1]
    algs.comment('Estimation: p_V_C[:,i] = algs.twist2HomogMatrix(T_W_C[:,i])[:3,-1]')
    
    if 1:
        plt.scatter(pp_G_C[2], -pp_G_C[0], marker='.', label='Groud Truth') 
        plt.scatter(p_V_C[2],-p_V_C[0], marker='.', label='Estimate') 
        plt.axis('equal')
        plt.legend() 
        plt.title('Compare estimated trajectory to ground truth')
        plt.show()
        print('Ground Truth and Estimation is shown') 

    # ---------------------------------------------------
    # PART 1: TRAJECTORY ALIGNMENT  
    # ---------------------------------------------------
    algs.title('part 1: trajectory alignment')

    p_G_C = algs.alignEstimateToGroundTruth(pp_G_C, p_V_C)

    if 1:
        print('pp_G_C.shape: ', pp_G_C.shape) 
        print('p_V_C.shape: ', p_V_C.shape)
        print('p_G_C.shape: ', p_G_C.shape) 

    if 1:
        algs.comment('Aligned: p_G_C = algs.alignEstimateToGroundTruth(pp_G_C, p_V_C)')
        plt.scatter(pp_G_C[2], -pp_G_C[0], marker='.', label='Groud Truth') 
        plt.scatter(p_V_C[2], -p_V_C[0], marker='.', label='Estimate') 
        plt.scatter(p_G_C[2], -p_G_C[0], marker='.', label='Aligned')
        plt.axis('equal')
        plt.legend() 
        plt.show()

    # ---------------------------------------------------
    # PART 2: small bundle adjustment   
    # ---------------------------------------------------

    if 0:
        algs.title('part 2: small bundle adjustment')

        # ---------------------------------------------------
        # x: the frame poses and landmark positions 
        # Y: the 2D coorcinates of each landmark observation
        # f(x): the 2D coordinates that should be observed acoording to the model defined by x. 
        # ---------------------------------------------------

        #hidden_state, observations, pp_G_C = algs.cropProblem(hidden_state, observations, pp_G_C, 4)
        #cropped_hidde_state = algs.cropProblem(hidden_state, observations, K)
        hidden_state = algs.runBA(hidden_state, observations, K) 
        T_W_C = hidden_state[:6*num_frames].reshape(-1,6).T # the camera pose of each frame in twist 
        P_lmks = hidden_state[-3*num_lmks:].reshape(-1,3).T # the 3D positions of landmarks 


if __name__=="__main__":
    main() 
