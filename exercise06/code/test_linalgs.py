import numpy as np 

def run_test_linalgs(homogenous, vec, unvec, krons):
    N = 4
    m = 8

    F = np.random.randint(1,m,(3,3)) 

    p1 = np.random.randint(1,m,(2,N))
    p2 = np.random.randint(1,m,(2,N)) 

    if 1: 
        print('*check the initital parameters*') 
        print('\n NumPoints=', N)
        print('\n F= \n', F)
        print('\n p1='); print(p1) 
        print('\n p2='); print(p2)
        print('\n') 
    
    if 1:
        print('*check vectrization and unvectrization*')
        print('\n flatten F={}'.format(F.flatten()))
        print('\n vec(F) = {}'.format(vec(F))) 
        print('\n unvec(vec(F)) = \n {}'.format(unvec(vec(F))))
        print('\n')

    p1 = homogenous(p1)
    p2 = homogenous(p2)

    if 1:
        print('*check homogenous*')
        print('\n p1='); print(p1) 
        print('\n p2='); print(p2)
        print('\n') 
    
    if 0: # check np.kron(a, b) 
        i = 1
        print('i = %d' % i)
        print('p2.T[:,i] @ F @ p1[:,i] = %d' % (p2[:,i].T @ F @ p1[:,i]))
        print('np.kron(p1[:,i], p2[:,i]).T @ vec(F)= %d \n' % (np.kron(p1[:,i], p2[:,i]).T @ vec(F)))

        print('np.kron(p1[:,i], p2[:,i]).T=', np.kron(p1[:,i], p2[:,i]).T)
        print('np.kron(p1, p2).T=\n', np.kron(p1, p2).T)
        print('\n')

    
    Q = krons(p1, p2) 

    print('*check krons*')
    print('\n Q ='); print(Q)

    print('\n Q @ vec(F) = ', Q @ vec(F)) 
    print('\n p2[:,i].T @ F @ p1[:,i] = ', [p2[:,i].T @ F @ p1[:,i] for i in range(p1.shape[1])])
    print('\n diff = ', Q @ vec(F) - np.array( [p2[:,i].T @ F @ p1[:,i] for i in range(p1.shape[1])]))
 
from linalgs import * 
if __name__=="__main__":
    run_test_basics(homogenous, vec, unvec, krons)
