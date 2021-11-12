def calc_ncc_fast(gray_left, gray_right, patch_radius=1):
    m_height, m_width = gray_left.shape 
    new_height, new_width = m_height - 2*patch_radius, m_width - 2*patch_radius 

    patches_left = np.zeros((m_height, m_width, (2*patch_radius+1)**2), dtype=np.uint8)
    patches_right = np.zeros_like(patches_left) # data type also copied 
    for i in range(2*patch_radius+1):
        for j in range(2*patch_radius+1):
            patches_left[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_left.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
            patches_right[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_right.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
    patches_left = patches_left[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]
    patches_right = patches_right[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]
    
    # TODO: normalize each patch
    #print(patches_right.dtype) # uint8 
    patches_left = normalize(patches_left) 
    patches_right = normalize(patches_right) 
    #print(patches_right.dtype) # float64 
    
    # TODO: Compute correlation.
    # Hint: This can be computed as a matrix multiplication. Check np.matmul and np.transpose     
    ncc = patches_left @ np.transpose(patches_right, axes=(0,2,1)) 
    
    return ncc 
