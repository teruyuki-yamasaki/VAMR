from scipy.spatial.distance import cdist 
def run_test_cdist():
    print('*'*5 + 'test on cdist' + '*'*5)

    n_dim = 5
    Ma = 4
    Mb = 3
    a = np.random.randint(0,10, (Ma, n_dim))
    b = np.random.randint(0,10, (Mb, n_dim)) 
    print(f'\n a{a.shape} = \n', a)
    print(f'\n b{b.shape} = \n', b) 

    for i in range(Ma):
        row = a[i].reshape(1,-1) - b 
        print(f'\n a{i} - b = \n', row) 

        print('\n *^2 = \n', row**2) 

        print(f'\n sum(*^2) = \n', np.sum(row**2, axis=-1, keepdims=False))

        print(f'\n sqrt(sum(*^2)) = \n', np.sqrt(np.sum(row**2, axis=-1, keepdims=False)))

    measure = 'euclidean'
    dist = cdist(a, b, measure) 
    print(f'\n dist{dist.shape, measure} = \n', dist) 

if __name__=="__main__":
    run_test_cdist()
