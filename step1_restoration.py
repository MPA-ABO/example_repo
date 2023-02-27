import napari
import numpy as np

from load_data import load_data
from evaluate import PSNR, SSIM

from scipy.ndimage import gaussian_filter

def step1_restoration(data_tzxy_np, info):
    '''
    This is function for preprocessing of 4D data. It should remove background and noise.
    
    Parameters
    ----------
    data_tzxy_np : numpy array with np.float64 dtype, axis order tzxy
        4D MRI data of heard.
    info : any data type
        custom set of required values and metadata.

    Returns
    -------
    data_tzxy_np_restored : numpy array with np.float64 dtype, axis order tzxy (same as input)
        preprocessed 4D data without background and noise.
    info : any data type
        custom set of required values required for next step and metadata.
    '''
    
    data_tzxy_np_restored = np.zeros_like(data_tzxy_np)
    for t in range(data_tzxy_np.shape[0]):
        data_tzxy_np_restored[t, ...] = gaussian_filter(data_tzxy_np[t, ...], [0.3, 1, 1]) - gaussian_filter(data_tzxy_np[t, ...], [4, 15, 15])
    
    return data_tzxy_np_restored, info



if __name__ == "__main__":
    
    data_tzxy_np, info  = load_data('../DataABO/patient061/patient061_4d_noisy.nii.gz')
    data_tzxy_np_gt, info  = load_data('../DataABO/patient061/patient061_4d.nii.gz')
    
    data_tzxy_np_restored, info  = step1_restoration(data_tzxy_np, info)
    
    print('PSNR: ' + str(PSNR(data_tzxy_np_gt, data_tzxy_np_restored)))
    print('SSIM: ' + str(SSIM(data_tzxy_np_gt, data_tzxy_np_restored)))
    
    viewer = napari.view_image(np.stack((data_tzxy_np, data_tzxy_np_restored, data_tzxy_np_gt), axis=0))
    napari.run()