import napari
import numpy as np
import yaml
import os


from evaluate import DICE, hausdorff
from evaluate import read2np
from load_data import load_data
from scipy.ndimage import label

def step2_segmentation_lv(data_tzxy_np_restored, info):
    '''
    This is function segmentation .... It should also detect systolic and diastolic 
    frame and return those frames and coresponding segmentattion masks.

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
        3D data at time of detected end-systole.
    ED_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle (LV) at time of detected end-diastole - coresponding to ED_data_zxy_np.
    ES_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle (LV) at time of detected end-systole - coresponding to ES_data_zxy_np.
    info : any data type
        custom set of required values required for next step and metadata.
    '''
    
    
    ED_data_zxy_np = data_tzxy_np_restored[0,...]
    ES_data_zxy_np = data_tzxy_np_restored[-5,...]
    
    ED_mask_lv_zxy_np = label(ED_data_zxy_np > 200)[0] == 52
    ES_mask_lv_zxy_np = label(ES_data_zxy_np > 200)[0] == 57
    
    return ED_data_zxy_np, ES_data_zxy_np, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, info



if __name__ == "__main__":
    
    fname = '../DataABO/patient061/patient061_4d_noisy.nii.gz'
    
    data_tzxy_np, info  = load_data(fname)
    
    with open(os.path.split(fname)[0] + '/Info.cfg') as f:
        info_cfg = yaml.safe_load(f)
        
    data_tzxy_np_restored_gt = read2np(fname.replace('4d_noisy.nii.gz', '4d.nii.gz'))
    ED_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '.nii.gz')
    ES_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '.nii.gz')
    ED_mask_lv_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 3
    ES_mask_lv_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz') == 3
    
    ED_data_zxy_np, ES_data_zxy_np, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, info = step2_segmentation_lv(data_tzxy_np_restored_gt, info)
     
    v = napari.Viewer()
    datalayer = v.add_image(ES_data_zxy_np, name='data')
    datalayer.blending = 'additive'
    datalayer.colormap = 'gray'
    gtlayer = v.add_image(ES_mask_lv_zxy_np_gt, name='gt')
    gtlayer.colormap = 'green'
    gtlayer.blending = 'additive'
    reslayer = v.add_image(ES_mask_lv_zxy_np, name='resES')
    reslayer.colormap = 'magenta'
    reslayer.blending = 'additive'
    napari.run()
    
    
    print('DICE: ' + str(DICE(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np_gt)))
    print('Haussdorff: ' + str(hausdorff(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np_gt)))
    
    
    
    
    
    
    