import yaml
import os

import numpy as np

from load_data import load_data
from evaluate import AE, read2np, get_resolution

def step4_analysis(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_myo_zxy_np, info=None):
    '''
    This function is for extraction of required values for analysis of heart.
    It shoud measure ejection fraction and myocard weight. It can use semgentation masks 
    created in previous steps.

    Parameters
    ----------
    ED_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle at time of detected end-diastole - corresponding to ED_data_zxy_np.
    ES_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle at time of detected end-systole - corresponding to ES_data_zxy_np.
    ED_mask_myo_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of myocard at time of detected end-diastole - corresponding to ED_data_zxy_np.
    info : any data type
        custom set of required values and metadata.

    Returns
    -------
    ejection_fraction : float
        estimated ejection fraction (%).
    myocard_weight : float
        estimated myocard's weight (g).

    '''
    
    
    ejection_fraction = 60.0 # (%)
    myocard_weight = 100.0 # (g)
    
    return ejection_fraction, myocard_weight



if __name__ == "__main__":
    
    fname = '../DataABO/patient061/patient061_4d_noisy.nii.gz'
    
    data_tzxy_np, info  = load_data(fname)
    
    with open(os.path.split(fname)[0] + '/Info.cfg') as f:
        info_cfg = yaml.safe_load(f)
        
    ED_mask_lv_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 3
    ES_mask_lv_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz') == 3
    ED_mask_myo_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 2
    ED_resolution_gt = get_resolution(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz')
    
    ejection_fraction_gt = ((np.sum(ED_mask_lv_zxy_np_gt) - np.sum(ES_mask_lv_zxy_np_gt)) / np.sum(ED_mask_lv_zxy_np_gt)) * 100 # (%)
    rho = 1.05 / 1000 # average volumetric mass density of myocard (g/mm^3)
    voxel_volume = ED_resolution_gt[0] * ED_resolution_gt[1] * ED_resolution_gt[2]  # mm^3
    myocard_weight_gt = np.sum(ED_mask_myo_zxy_np_gt) * voxel_volume * rho
    
    ejection_fraction, myocard_weight = step4_analysis(ED_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np_gt, ED_mask_myo_zxy_np_gt, info)
    
    print('Absoulte Error - Ejection Fraction: ' + str(AE(ejection_fraction_gt, ejection_fraction)) + ' %')
    print('Absoulte Error - Myocardial Weight: ' + str(AE(myocard_weight_gt, myocard_weight)) + ' g')