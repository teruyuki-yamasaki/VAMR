import numpy as np 
import matplotlib.pyplot as plt 

def generateLinearData(args, ans=False):
    coef = np.random.rand(2) 

    # determin x range 
    xspan = args['xspan']
    x_lim = np.ones(2) * args['xstart']
    x_lim[1] += xspan

    # determine y range 
    y_lim = np.polyval(coef, x_lim) 
    yspan = np.abs(y_lim[1]-y_lim[0]) 

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

def ransacLinear(data, params):
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
        n = np.random.choice(np.arange(numPoints), 2) # note that we must not allow duplication 
        p0 = data[:,n[0]]
        p1 = data[:,n[1]] 
        

        # fitting based on maybeInliers 
        d = p1 -  p0 
        theta = np.arctan2(d[1], d[0])  
        ed = np.array([np.cos(theta), np.sin(theta)]) 
        en = np.array([[0,-1],[1,0]]) @ ed 

        # compute the distances of all other points from this line 
        dist = np.abs((data-p0.reshape(2,-1)).T @ en)
        countInliers = np.count_nonzero(dist<max_noise)

        if countInliers > numPoints * inlierRatio:
            inliers = data.T[dist<max_noise]
            xin = inliers[:,0]
            yin = inliers[:,1] 
            coef_in, thisErr, _, _, _ = np.polyfit(xin, yin, 1, full=True)
            
            if thisErr < bestErr:
                bestFit = coef_in 
        
        numIterations += 1
        
    return bestFit 
 
def test_ransacLinear(generator, ransac=None):
    data, max_noise, Dgt, Din, Dout = generateLinearData(generator, ans=True)  

    fig, axs = plt.subplots(1, 2, figsize=(20, 5)) 

    axs[0].set_title('given data')
    axs[0].scatter(data[0],data[1])

    axs[1].set_title('results')
    axs[1].plot(Dgt[0], Dgt[1], color='green', label='ground truth')
    axs[1].scatter(Din[0], Din[1], color='blue', label='inliers')
    axs[1].scatter(Dout[0], Dout[1], color='red', label='outliers') 

    if ransac!=None:
        ransac['max_noise'] = max_noise
        guess = ransacLinear(data, ransac)
        axs[1].plot(Dgt[0], np.polyval(guess, Dgt[0]), color='orange', label='best guess with ransac')

    plt.legend()  
    plt.show() 
 
if __name__=="__main__":
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
  
  test(generator, ransac)  
