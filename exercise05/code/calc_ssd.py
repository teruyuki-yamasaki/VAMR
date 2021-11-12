def calc_ssd(gray_left, gray_right, patch_radius):
    m_height, m_width = gray_left.shape 
    new_height, new_width = m_height - 2*patch_radius, m_width - 2*patch_radius 

    # create patches 
    patches_left = np.zeros((m_height, m_width, (2*patch_radius+1)**2), dtype=np.uint8) # 画素値なので0-255:uint8 
    patches_right = np.zeros_like(patches_left)
    for i in range(2*patch_radius+1):
        for j in range(2*patch_radius+1):
            patches_left[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_left.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
            patches_right[:,:,(2*patch_radius+1)*i + j] = np.roll(gray_right.flatten(), -(m_width*i+j)).reshape(m_height, m_width)
    patches_left = patches_left[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]
    patches_right = patches_right[:m_height - 2*patch_radius, :m_width - 2*patch_radius, :]

    ssd = np.zeros((new_height, new_width, new_width), dtype=np.uint32) # 255**2 = 65025 < を余裕でおさめたい: uint16 ~ uin32
    for y in range(new_height):
        for x_l in range(new_width):
            diff = patches_left[y, x_l,:] - patches_right[y,:,:]
            ssd[y, x_l, :] = np.sum(diff**2,axis=1)

            if 0:
                print(patches_left[y, x_l,:].shape)
                print("L=",patches_left[y,x_l,:])
                print(patches_right[y,:,:].shape)
                print("R=",patches_right[y,:,:])
                print(diff.shape)
                print("diff=",diff)  
                print(np.sum(diff**2,axis=1).shape)
                print(np.sum(diff**2,axis=1))

    return ssd 
