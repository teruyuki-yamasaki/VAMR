def gradop(method='Sobel'):
    if method=='Sobel':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    elif method=='Prewitt':
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) 

def imgrad(img, method='Sobel', alg=1):
    height, width = img.shape 
    op = gradop(method) 
    kernel_radius = op.shape[0] // 2
    dI = np.zeros((height-2*kernel_radius, width-2*kernel_radius, 2)) 

    #replace this with a faster algorithm 
    if alg==0: 
        opT = op.T 
        for y in range(dI.shape[0]):
            for x in range(dI.shape[1]): 
                I = img[y:y+2*kernel_radius+1, x:x+2*kernel_radius+1]  
                dI[y, x, 0] = np.sum(I * op)
                dI[y, x, 1] = np.sum(I * opT)
    elif alg==1: #soma = the state of my art 
        opT = op.T 
        patches = np.zeros((height, width, (2*kernel_radius+1)**2))
        for y in range(2*kernel_radius+1):
            for x in range(2*kernel_radius+1):
                patches[:,:,(2*kernel_radius+1)*y+x] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width)
        patches = patches[:height-2*kernel_radius, :width-2*kernel_radius, :]

        dI[:,:,0] = np.sum(patches[:,:,:] * op.flatten(), axis=2)
        dI[:,:,1] = np.sum(patches[:,:,:] * opT.flatten(), axis=2)
    
    elif alg==2: #結果が合わない＆遅い?
        patches = np.zeros((height, width, (2*kernel_radius+1)**2, 2))
        for y in range(2*kernel_radius+1):
            for x in range(2*kernel_radius+1):
                patches[:,:,(2*kernel_radius+1)*y+x,0] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width) * op.flatten()[(2*kernel_radius+1)*y+x]
                patches[:,:,(2*kernel_radius+1)*y+x,1] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width) * op.T.flatten()[(2*kernel_radius+1)*y+x]
        patches = patches[:height-2*kernel_radius, :width-2*kernel_radius, :,:]
        dI = np.sum(patches, axis=2)

    return dI

t0 = time.time() 
dI = ImageGradientXY(img, alg=0) 
t1 = time.time() 
print('{:.2f}'.format(t1-t0)) #7.04
imshow(dI[:,:,0], 'Ix') 

t0 = time.time() 
dI = ImageGradientXY(img, alg=1) 
t1 = time.time() 
print('{:.2f}'.format(t1-t0))
imshow(dI[:,:,0], 'Ix') #0.11

t0 = time.time() 
dI = ImageGradientXY(img, alg=2) 
t1 = time.time() 
print('{:.2f}'.format(t1-t0))
imshow(dI[:,:,0], 'Ix') #0.28
