import numpy as np 
import matplotlib.pyplot as plt 

def generateParabolicData(args, ans=False):
    coef = np.random.rand(3) 

    # determin x range 
    xspan = args['xspan']
    x_lim = np.ones(2) * args['xstart']
    x_lim[1] += xspan

    # determine y range
    xc = - coef[1]/(2*coef[0]) 
    yc = np.polyval(coef,xc)
    y_lim = np.polyval(coef, x_lim) 
    id = np.argmin([np.abs(y_lim[0]-yc), np.abs(y_lim[1]-yc)])  
    y_lim[id] = yc 
    yspan = np.abs(y_lim[1] - y_lim[0])

    # determine noise range 
    max_noise = yspan * args['noise_ratio'] 

    # genetrate ground truth 
    xtruth = np.linspace(x_lim[0], x_lim[1], 1000)
    ytruth = np.polyval(coef, xtruth) 
    

    # generate inliers 
    xin = np.random.rand(args['num_inliers'])*xspan + x_lim[0]
    yin = np.polyval(coef, xin) 
    yin += (np.random.rand(yin.shape[0])-.5) * 2 * max_noise
    

    # generate outliers 
    xout = np.random.rand(args['num_outliers'])*xspan + x_lim[0] 
    yout = np.random.rand(args['num_outliers'])*yspan + np.min(y_lim) 

    # put them together 
    data = np.concatenate([xin, xout, yin, yout]).reshape(2,-1)

    if ans==False:
        return data, max_noise 

    else:
        Dgt = np.concatenate([xtruth, ytruth]).reshape(2,-1) 
        Din = np.concatenate([xin, yin]).reshape(2,-1)
        Dout = np.concatenate([xout, yout]).reshape(2,-1)
        return data, max_noise, Dgt, Din, Dout 

def ransacParabola(data, params):
    maxIterations = params['maxIterations']
    inlierRatio = params['inlierRatio']
    max_noise = params['max_noise'] 

    numPoints = data.shape[1] 
    numIterations = 0
    bestFit = None 
    bestErr = np.inf 

    while numIterations < maxIterations:
        # maybeInliers: n randomly selected values from dadta 
        # randomly select a sample of 2 points from A 
        n = np.random.choice(np.arange(numPoints), 3) # note that we must not allow duplication 
        p0 = data[:,n[0]]
        p1 = data[:,n[1]] 
        p2 = data[:,n[2]]
        
        # fitting based on maybeInliers 
        ps = np.concatenate([p0, p1, p2]).reshape(-1,2) 
        coef = np.polyfit(ps[:,0],ps[:,1],2) 

        # compute the distances of all other points from this line 
        dist = np.abs(data[1] - np.polyval(coef, data[0]))
        countInliers = np.count_nonzero(dist<max_noise)

        if countInliers > numPoints * inlierRatio:
            inliers = data.T[dist<max_noise]
            xin = inliers[:,0]
            yin = inliers[:,1] 
            coef_in, thisErr, _, _, _ = np.polyfit(xin, yin, 2, full=True)
            
            if thisErr < bestErr:
                bestFit = coef_in 
        
        numIterations += 1
        
    return bestFit 

def test_ransacParabola(generator, ransac=None):
        data, max_noise, Dgt, Din, Dout = generateParabolicData(generator, ans=True)  
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 5)) 

        axs[0].set_title('given data')
        axs[0].scatter(data[0],data[1])

        axs[1].set_title('results')
        axs[1].plot(Dgt[0], Dgt[1], color='green', label='ground truth')
        axs[1].scatter(Din[0], Din[1], color='blue', label='inliers')
        axs[1].scatter(Dout[0], Dout[1], color='red', label='outliers') 

        fullfit = np.polyfit(data[0],data[1],2)
        axs[1].plot(Dgt[0], np.polyval(fullfit, Dgt[0]), color='pink', label='full fit')

        if ransac!=None:
            ransac['max_noise'] = max_noise

            for i in range(3):
                guess = ransacParabola(data, ransac)
                axs[1].plot(Dgt[0], np.polyval(guess, Dgt[0]), color='orange', label='best guesses with ransac' if i==0 else '')
                
        plt.legend()  
        plt.show() 

generator = {
    'num_inliers': 20,
    'num_outliers': 10,
    'noise_ratio': 0.1,
    'xstart': 0,
    'xspan': 1,
}

ransac = {
    'maxIterations': 16,
    'inlierRatio': 0.3
}

test_ransacParabola(generator, ransac)
