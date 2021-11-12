min_disp = 5
max_disp = 50

def disparity(gray_left, gray_right, patch_radius=5, measure='ncc'):
    if measure=='ncc':
        ncc = calc_ncc_fast(gray_left, gray_right, patch_radius) 
        ncc = np.tril(ncc, k=-1) 
        
        x_right = np.argmax(ncc, axis=2).astype(int)
        x_left = (np.ones(ncc.shape[0],dtype=int) * np.arange(ncc.shape[1]).reshape(-1,1)).T.astype(int) 
        
        d = x_left - x_right
        d = (d - np.min(d)) / (np.max(d) - np.min(d)) * 255 
        d = d.astype(np.uint8)
    
    elif measure=='ssd':
        ssd = calc_ssd(gray_left, gray_right, patch_radius) 
        ssd = np.triu(np.ones_like(ssd), k=-1) * np.max(ssd) + np.tril(ssd, k=-1)
        
        x_right = (np.argmin(ssd, axis=2)).astype(np.uint16)        
        x_left = ((np.ones(ssd.shape[0]) * np.arange(ssd.shape[1]).reshape(-1,1)).T).astype(np.uint16)
        
        d = x_left - x_right

        ###TODO: put lower and upper bounds to d 
        d = d * (min_disp<=d) * (d<=max_disp)

        # negative disparity 
        d = np.max(d) - d
    
    return d 
