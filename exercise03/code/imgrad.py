import numpy as np 

def impad(img, pad=(1,1)):
    img_pad = np.zeros((len(img)+2*pad[0], len(img[0])+2*pad[1]), dtype=float)
    img_pad[pad[0]:-pad[0], pad[1]:-pad[1]] = img 
    return img_pad 

def gradop(method='Sobel'):
    if method=='Sobel':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)

    elif method=='Prewitt':
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int8)  

def imgrad(img, method='Sobel', alg=1): 
    height, width = img.shape 
    op = gradop(method)   
    kernel_radius = op.shape[0] // 2
    dI = np.zeros((height-2*kernel_radius, width-2*kernel_radius, 2), dtype=np.int16) 

    #replace this with a faster algorithm 
    if alg==0: # 7.13
        opT = op.T 
        for y in range(dI.shape[0]):
            for x in range(dI.shape[1]): 
                I = img[y:y+2*kernel_radius+1, x:x+2*kernel_radius+1]  
                dI[y, x, 0] = np.sum(I * op)
                dI[y, x, 1] = np.sum(I * opT)

    elif alg==1: # 0.07
        opT = op.T 
        patches = np.zeros((height, width, (2*kernel_radius+1)**2), dtype=np.uint16)
        for y in range(2*kernel_radius+1):
            for x in range(2*kernel_radius+1):
                patches[:,:,(2*kernel_radius+1)*y+x] = np.roll(img.flatten(), -(width*y+x)).reshape(height, width)
        patches = patches[:height-2*kernel_radius, :width-2*kernel_radius, :]

        dI[:,:,0] = np.sum(patches[:,:,:] * op.flatten(), axis=-1)
        dI[:,:,1] = np.sum(patches[:,:,:] * opT.flatten(), axis=-1)

    return dI
