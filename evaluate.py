import glob
import SimpleITK as sitk
import numpy as np
import yaml
import os

from load_data import load_data
from step1_restoration import step1_restoration
from step2_segmentation import step2_segmentation
from step3_anylysis import step3_anylysis


def read2np(fname):
    
    return sitk.GetArrayFromImage(sitk.ReadImage(fname))


if __name__ == "__main__":
    
    data_path = '../DataABO'
    
    fnames = glob.glob(data_path + '/**/*_4d_noisy.nii.gz')
    
    
    for fnum, fname in enumerate(fnames):
        
        
        with open(os.path.split(fname)[0] + '/Info.cfg') as f:
            info_cfg = yaml.safe_load(f)
        
        
        data_tzxy_np_restored_gt = read2np(fname.replace('4d_noisy.nii.gz', '4d.nii.gz'))
        ED_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '.nii.gz')
        ES_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '.nii.gz')
        ED_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz')
        ES_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz')
        ejection_fraction_gt = 0.3##########################################################################################################################!!!!!!
        myocard_weight_gt = info_cfg['Weight']
        

        data_tzxy_np, info  = load_data(fname)
        
        
        data_tzxy_np_restored, info = step1_restoration(data_tzxy_np, info)
        
        ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info = step2_segmentation(data_tzxy_np_restored, info)
        
        ejection_fraction, myocard_volume = step3_anylysis(ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info)