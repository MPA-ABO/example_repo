import napari
import yaml
import os


from evaluate import DICE, hausdorff
from evaluate import read2np
from load_data import load_data
from scipy.ndimage import label
from skimage.morphology import binary_dilation, disk

def step3_segmentation_myo(ED_data_zxy_np, ED_mask_lv_zxy_np, info):
    '''
    This is function for segmentation of myocard in end-diastolic frame.
    It should return corresponding segmentation mask.

    Parameters
    ----------
    ED_data_zxy_np : numpy array with np.float64 dtype, axis order zxy
        3D data at time of detected end-diastole.
    ED_mask_lv_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of left ventricle (LV) at time of detected end-diastole - corresponding to ED_data_zxy_np.
    info : any data type
        custom set of required values and metadata.

    Returns
    -------
    ED_mask_myo_zxy_np : numpy array with bool dtype, axis order zxy
        3D segmentation mask of myocard at time of detected end-diastole - corresponding to ED_data_zxy_np.
    info : any data type
        custom set of required values required for next step and metadata.
    '''
    
    strel = disk(7)

    ED_mask_myo_zxy_np = ED_mask_lv_zxy_np.copy()
    for z in range(ED_mask_myo_zxy_np.shape[0]):
        ED_mask_myo_zxy_np[z, ...] = binary_dilation(ED_mask_myo_zxy_np[z, ...], strel)
    
    ED_mask_myo_zxy_np[ED_mask_lv_zxy_np] = 0
    
    return ED_mask_myo_zxy_np, info



if __name__ == "__main__":
    
    fname = '../DataABO/patient061/patient061_4d_noisy.nii.gz'
    
    data_tzxy_np, info  = load_data(fname)
    
    with open(os.path.split(fname)[0] + '/Info.cfg') as f:
        info_cfg = yaml.safe_load(f)
        
    ED_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '.nii.gz')
    ED_mask_lv_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 3
    ED_mask_myo_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 2
    
    ED_mask_myo_zxy_np, info = step3_segmentation_myo(ED_data_zxy_np_gt, ED_mask_lv_zxy_np_gt, info)
     
    v = napari.Viewer()
    datalayer = v.add_image(ED_data_zxy_np_gt, name='data')
    datalayer.blending = 'additive'
    datalayer.colormap = 'gray'
    gtlayer = v.add_image(ED_mask_myo_zxy_np_gt, name='gt')
    gtlayer.colormap = 'green'
    gtlayer.blending = 'additive'
    reslayer = v.add_image(label(ED_mask_myo_zxy_np)[0], name='resED')
    reslayer.colormap = 'magenta'
    reslayer.blending = 'additive'
    napari.run()
    
    
    print('Dice coefficient: ' + str(DICE(ED_mask_myo_zxy_np, ED_mask_myo_zxy_np_gt)))
    print('Hausdorff distance: ' + str(hausdorff(ED_mask_myo_zxy_np, ED_mask_myo_zxy_np_gt)))
    