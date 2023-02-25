import napari
import numpy as np

from load_data import load_data

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
    
    data_tzxy_np_restored = 0
    
    
    return data_tzxy_np_restored, info


if __name__ == "__main__":
    
    
    data_tzxy_np, info  = load_data('../DataABO/patient061/patient061_4d.nii.gz')
    
    data_tzxy_np_restored, info  = step1_restoration(data_tzxy_np, info)
    
    viewer = napari.view_image(np.stack((data_tzxy_np, data_tzxy_np_restored), axis=0))
    napari.run()