import numpy as np 

def StructureTensorM(dI, patch_radius=1): # 11.20 # なぜ結果が変わった？
    height, width, _ = dI.shape

    M = np.zeros((height-2*patch_radius, width-2*patch_radius, 3))  
    for y in range(M.shape[0]):
        for x in range(M.shape[1]): 
            Ix = dI[y:y+2*patch_radius+1, x:x+2*patch_radius+1,0]  
            Iy = dI[y:y+2*patch_radius+1, x:x+2*patch_radius+1,1] 
            M[y,x,:] = [np.sum(Ix**2), np.sum(Ix*Iy), np.sum(Iy**2)]

    return M

def StructureTensorM_fast(dI, patch_radius=1): # 0.65
    height, width, _ = dI.shape

    Ix = np.zeros((height, width, (2*patch_radius+1)**2), dtype=np.int32)
    Iy = np.zeros((height, width, (2*patch_radius+1)**2), dtype=np.int32) 
    for y in range(2*patch_radius+1):
        for x in range(2*patch_radius+1):
            Ix[:,:,(2*patch_radius+1)*y + x] = np.roll(dI[:,:,0].flatten(), -(width*y+x)).reshape(height, width)
            Iy[:,:,(2*patch_radius+1)*y + x] = np.roll(dI[:,:,1].flatten(), -(width*y+x)).reshape(height, width)
    Ix = Ix[:height - 2*patch_radius, :width - 2*patch_radius, :]
    Iy = Iy[:height - 2*patch_radius, :width - 2*patch_radius, :]

    M = np.zeros((height-2*patch_radius, width-2*patch_radius, 3), np.int32)
    M[:,:,0] = np.sum(Ix**2, axis=-1)
    M[:,:,1] = np.sum(Ix*Iy, axis=-1)
    M[:,:,2] = np.sum(Iy**2, axis=-1) 
    
    return M
