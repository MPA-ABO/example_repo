

def step4_analysis(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_myo_zxy_np, info):
    '''
    This function is for extraction of required valeus for analysis of heart.
    It shoud measure ejection fraction and myocard_volume. It can used semgentation mask 
    created in previous step.

    Parameters
    ----------
    ED_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle at time of detected end-diastole - coresponding to ED_data_zxy_np.
    ES_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle at time of detected end-systole - coresponding to ES_data_zxy_np.
    ED_mask_myo_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of myocardium at time of detected end-diastole - coresponding to ED_data_zxy_np.
    info : any data type
        custom set of required values  and metadata.

    Returns
    -------
    ejection_fraction : float
        estimated ejection fraction (%).
    myocard_weight : float
        estimated myocard weight (g).

    '''
    
    
    ejection_fraction = 60.0 # (%)
    myocard_weight = 100.0 # (g)
    
    return ejection_fraction, myocard_weight

