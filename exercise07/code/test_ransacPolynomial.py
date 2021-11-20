import numpy as np 
import matplotlib.pyplot as plt 

def test_ransacPoly(args, deg=1, ransac=None):
    # generate data 
    data, max_noise, Dgt, Din, Dout = generatePolyData(args, deg)  
    
    # show data 
    fig, axs = plt.subplots(1, 2, figsize=(20, 5)) 

    # show generated data 
    axs[0].set_title('given data')
    axs[0].scatter(data[0],data[1])

    # show ground truth and predictions 

    # ground truth 
    axs[1].set_title('results')
    axs[1].plot(Dgt[0], Dgt[1], color='green', label='ground truth')
    axs[1].scatter(Din[0], Din[1], color='blue', label='inliers')
    axs[1].scatter(Dout[0], Dout[1], color='red', label='outliers') 

    # predictions 
    # fullfit 
    fullfit = np.polyfit(data[0], data[1], deg)
    axs[1].plot(Dgt[0], np.polyval(fullfit, Dgt[0]), color='pink', label='full fit')

    # ransac 
    if ransac!=None:
        args['max_noise'] = max_noise
        for i in range(3):
            guess = ransacPoly(data, args, deg)
            axs[1].plot(Dgt[0], np.polyval(guess, Dgt[0]), color='orange', label='best guesses with ransac' if i==0 else '')
            
    plt.legend()  
    plt.show() 

def generatePolyData(args, deg):
    coef = np.random.rand(deg+1)  

    # determin x range 
    xspan = args['xspan']
    x_lim = np.ones(2) * args['xstart']
    x_lim[1] += xspan

    # genetrate ground truth 
    xtruth = np.linspace(x_lim[0], x_lim[1], 1000)
    ytruth = np.polyval(coef, xtruth) 

    # determine y range 
    y_lim = [np.min(ytruth), np.max(ytruth)]
    yspan = y_lim[1] - y_lim[0]

    # determine noise range 
    max_noise = yspan * args['noise_ratio'] 

    # generate inliers 
    xin = np.random.rand(args['num_inliers'])*xspan + x_lim[0]
    yin = np.polyval(coef, xin) 
    yin += (np.random.rand(yin.shape[0])-.5) * 2 * max_noise

    # generate outliers 
    xout = np.random.rand(args['num_outliers'])*xspan + x_lim[0] 
    yout = np.random.rand(args['num_outliers'])*yspan + np.min(y_lim) 

    # put them together 
    data = np.concatenate([xin, xout, yin, yout]).reshape(2,-1)

    Dgt = np.concatenate([xtruth, ytruth]).reshape(2,-1) 
    Din = np.concatenate([xin, yin]).reshape(2,-1)
    Dout = np.concatenate([xout, yout]).reshape(2,-1)

    return data, max_noise, Dgt, Din, Dout 

def ransacPoly(data, args, degree):
    maxIterations = args['maxIterations']
    inlierRatio = args['inlierRatio']
    max_noise = args['max_noise'] 

    numIterations = 0
    bestFit = None 
    bestErr = np.inf 

    while numIterations < maxIterations:
        dist = clacDistance(data, degree)   
        countInliers = np.count_nonzero(dist<max_noise)

        if countInliers > data.shape[1] * inlierRatio:
            inliers = data.T[dist<max_noise]
            xin = inliers[:,0]
            yin = inliers[:,1] 
            coef_in, thisErr, _, _, _ = np.polyfit(xin, yin, degree, full=True)
            
            if thisErr < bestErr:
                bestFit = coef_in 
        
        numIterations += 1
        
    return bestFit 

def clacDistance(data, degree=1): 
    # randomly select a sample of deg+1 points from data 
    n = np.random.choice(np.arange(data.shape[1]), degree+1) # note that we must not allow duplication 

    if degree==1:
        # selected points 
        p0 = data[:,n[0]]
        p1 = data[:,n[1]] 

        # fitting based on selected points 
        d = p1 -  p0 
        theta = np.arctan2(d[1], d[0])  
        ed = np.array([np.cos(theta), np.sin(theta)]) 
        en = np.array([[0,-1],[1,0]]) @ ed 

        # compute the distances of all other points from this line (Euclidean)
        dist = np.abs((data-p0.reshape(2,-1)).T @ en)
    
    elif degree==2:
        # selected points 
        p0 = data[:,n[0]]
        p1 = data[:,n[1]]
        p2 = data[:,n[2]] 

        # fitting based on selected points 
        P = np.vstack((p0,p1,p2)).T
        X = P[0] 
        Y = P[1] 
        A = np.vstack((X**2, X, np.ones(3))).T
        coef = np.linalg.pinv(A) @ Y

        # compute the distances of all other points from this hypothetical curve 
        dist = np.abs(data[1] - np.polyval(coef, data[0])) 
    
    else:
        N = n.shape[0]

        # selected points 
        P = np.zeros((2, N))
        for i in range(N):
            P[:,i] = data[:,n[i]]
        
        # fitting based on selected points 
        X = P[0]
        Y = P[1] 
        A = np.ones((N,N)) 
        for i in range(N-1):
            A[:,i] = X**(N-1-i)
        coef = np.linalg.pinv(A) @ Y 

        # compute the distances of all other points from this hypothetical curve 
        dist = np.abs(data[1] - np.polyval(coef, data[0])) 
    
    return dist 


if __name__=="__main__":
    args = {
        'num_inliers': 20,
        'num_outliers': 10,
        'noise_ratio': 0.1,
        'xstart': 0,
        'xspan': 5,
        'maxIterations': 16,
        'inlierRatio': 0.3
    }

    test_ransacPoly(args, deg=3, ransac=True)
