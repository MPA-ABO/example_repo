

def step3_anylysis(ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info):
    '''
    This function is for extraction of required valeus for analysis of heard.
    It shoud measure ejection fraction and myocard_volume. It can used semgentation mask 
    created in previous step.

    Parameters
    ----------
    ED_data_zxy_np : numpy array with np.float64 dtype, axis order zxy
        3D data at time of detected end-diastole.
    ES_data_zxy_np : numpy array with np.float64 dtype, axis order zxy
        3D data at time of detected end-sysstole.
    ED_mask_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask at time of detected end-diastole - coresponding to ED_data_zxy_np.
    ES_mask_zxy_np : numpy array with bool dtype, axis order zxy
        3D data at time of detected end-sysstole - coresponding to ED_data_zxy_np..
    info : any data type
        custom set of required values  and metadata.

    Returns
    -------
    ejection_fraction : float
        estimated ejection fraction.
    myocard_weight : float
        estimated myocard weight.

    '''
    
    
    ejection_fraction = 5
    myocard_weight = 3
    
    return ejection_fraction, myocard_weight

