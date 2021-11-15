import numpy as np 

N = 10 #;         % Number of 3-D points

# Test linear triangulation
#P = randn(4,N);  % Homogeneous coordinates of 3-D points
#P(3, :) = P(3, :) * 5 + 10;
#P(4, :) = 1;

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

p1 = (p1 / p1[2,:])
p2 = (p2 / p2[2,:])

def skew0(p):
    op = np.array([
        [[0,0,0],[0,0,1],[0,-1,0]],
        [[0,0,-1],[0,0,0],[1,0,0]],
        [[0,1,0],[-1,0,0],[0,0,0]]
    ])

    px = op @ p

    if 0:
        print(op.shape)
        print(p.shape)
        print(px) 
    return px

p1xM1_0 = skew0(p1[:,0]) @ M1
p2xM2_0 = skew0(p2[:,0]) @ M2
A0 = np.vstack((p1xM1_0, p2xM2_0))
#print(A0)
tAA0 = np.transpose(A0, axes=(1,0)) @ A0
#print(tAA0)
U0, S0, V0 = np.linalg.svd(tAA0)
print(V0)
P0 = V0[-1,:] / V0[-1,-1] 
print(P0)
print(P[:,0])

def skew(p):
    op = np.array([
        [[0,0,0],[0,0,1],[0,-1,0]],
        [[0,0,-1],[0,0,0],[1,0,0]],
        [[0,1,0],[-1,0,0],[0,0,0]]
    ])

    px = np.transpose(op @ p.T, axes=(0,2,1)).reshape(-1,3,3)

    if 0:
        print(op.shape)
        print(p.shape)
        print(px) 
    
    return px

#print(P)

p1xM1_0 = skew(p1[:,0].reshape(1,3)) @ M1
p2xM2_0 = skew(p2[:,0].reshape(1,3)) @ M2
#A0 = np.vstack((p1xM1_0, p2xM2_0), axis=1)
A0 = np.concatenate([p1xM1_0, p2xM2_0], axis=-2)
#print(A0)
tAA0 = np.transpose(A0, axes=(0,2,1)) @ A0
#print(tAA0)
U0, S0, V0 = np.linalg.svd(tAA0)
#print(V0)
P0 = V0[:,-1,:] / V0[:,-1,-1] 
print(P0)
print(P[:,0])
