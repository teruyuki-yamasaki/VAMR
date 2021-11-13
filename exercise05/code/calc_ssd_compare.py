def calc_ssd(gray_left, gray_right, patch_radius):
    m_height, m_width = gray_left.shape 
    new_height, new_width = m_height - 2*patch_radius, m_width - 2*patch_radius 

    # create patches 0.80
    t0 = time.time()
    patches_left = np.zeros((m_height, m_width, (2*patch_radius+1)**2), dtype=np.uint8) # 画素値なので0-255:uint8 
    patches_right = np.zeros_like(patches_left)
    for i in range(2*patch_radius+1):
        for j in range(2*patch_radius+1):
            patches_left[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_left.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
            patches_right[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_right.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
    patches_left = patches_left[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]
    patches_right = patches_right[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]
    t1 = time.time() 
    print("create patches: {:.2f}".format(t1-t0))

    flag = 1
    print("flag: ", flag) 
    t0 = time.time()
    # for ループも適度に使った方が速い?? 
    if flag==0: # 55.16 54.21
        ssd = np.zeros((new_height, new_width, new_width), dtype=np.uint32) # 255**2 = 65025 < を余裕でおさめたい: uint16 ~ uin32
        for y in range(new_height):
            for x_l in range(new_width):
                diff = patches_left[y, x_l,:] - patches_right[y,:,:]
                ssd[y, x_l, :] = np.sum(diff**2,axis=1)
    
    elif flag==1: # 70.89 # 74.76
        left = patches_left.reshape(new_height, new_width, 1, (2*patch_radius+1)**2) # 0.00
        right = patches_right.reshape(new_height, 1, new_width, (2*patch_radius+1)**2) # 0.00 
        ssd = np.zeros((new_height, new_width, new_width), dtype=np.uint32) 
        for y in range(new_height):
            diff = left[y] - right[y] 
            ssd[y] = np.sum(diff**2, axis=-1) 

    else: # killed 
        left = patches_left.reshape(new_height, new_width, 1, (2*patch_radius+1)**2) # 0.00
        right = patches_right.reshape(new_height, 1, new_width, (2*patch_radius+1)**2) # 0.00 
        diff = left - right # 42.52
        ssd = np.sum(np.square(diff), axis=3) # killed 
    
    t1 = time.time() 
    print("calc ssd: {:.2f}".format(t1-t0))

    return ssd 
