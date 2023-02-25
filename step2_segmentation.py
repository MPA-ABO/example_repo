


def step2_segmentation(data_tzxy_np_restored, info):
    '''
    This is function segmentation .... It should also detect systolic and diastolic 
    frame and return those frames and coresponding segmentattion mask.

    Parameters
    ----------
    data_tzxy_np_restored : numpy array with np.float64 dtype, axis order tzxy
        preprocessed 4D MRI data of heard.
    info : any data type
        custom set of required values and metadata.

    Returns
    -------
    ED_data_zxy_np : numpy array with np.float64 dtype, axis order zxy
        3D data at time of detected end-diastole.
    ES_data_zxy_np : numpy array with np.float64 dtype, axis order zxy
        3D data at time of detected end-sysstole.
    ED_mask_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask at time of detected end-diastole - coresponding to ED_data_zxy_np.
    ES_mask_zxy_np : numpy array with bool dtype, axis order zxy
        3D data at time of detected end-sysstole - coresponding to ED_data_zxy_np.
    info : any data type
        custom set of required values required for next step and metadata.
    '''
    
    
    
    
    ED_data_zxy_np = 0
    ES_data_zxy_np = 0
    ED_mask_zxy_np = 0
    ES_mask_zxy_np = 0
    
    return ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info