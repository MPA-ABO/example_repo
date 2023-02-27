import napari
import numpy as np
import os
import yaml
import SimpleITK as sitk

from skimage.morphology import binary_dilation, disk
from evaluate import hausdorff


def read2np(fname):
    
    return sitk.GetArrayFromImage(sitk.ReadImage(fname))


fname = '../DataABO/patient061/patient061_4d_noisy.nii.gz'

with open(os.path.split(fname)[0] + '/Info.cfg') as f:
    info_cfg = yaml.safe_load(f)


ED_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz') == 3
ES_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz') == 3


strel = disk(5)

ED_mask_zxy_np = ED_mask_zxy_np_gt.copy()
ES_mask_zxy_np = ES_mask_zxy_np_gt.copy()
for z in range(ED_mask_zxy_np_gt.shape[0]):
    
    ED_mask_zxy_np[0, ...] = binary_dilation(ED_mask_zxy_np[0, ...], strel)
    ES_mask_zxy_np[0, ...] = binary_dilation(ES_mask_zxy_np[0, ...], strel)
    
    
print(hausdorff(ED_mask_zxy_np_gt, ES_mask_zxy_np_gt, ED_mask_zxy_np, ES_mask_zxy_np))
