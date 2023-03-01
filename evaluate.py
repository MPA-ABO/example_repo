import glob
import SimpleITK as sitk
import numpy as np
import yaml
import os
import json

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import hausdorff_distance

import pandas as pd



def PSNR(image_true, image_test):
    
    return peak_signal_noise_ratio(image_true, image_test, data_range=1000)



def SSIM(image_true, image_test):
    ssims = []
    for t in range(image_true.shape[0]):
        ssim = structural_similarity(image_true[0, ...], image_test[0, ...], data_range=1000, channel_axis=0)
        ssims.append(ssim)
    
    return np.mean(ssims)
    
  

def DICE(gt1, res1, gt2=None, res2=None):
    if gt2 is not None:
        gt = (np.stack((gt1, gt2)) > 0).astype(np.float32)
        seg = (np.stack((res1, res2)) > 0).astype(np.float32)
    else:
        gt = (gt1 > 0).astype(np.float32)
        seg = (res1 > 0).astype(np.float32)
    
    return np.sum(seg[gt == True])*2.0 / (np.sum(seg) + np.sum(gt))
    


def hausdorff(gt1, res1, gt2=None, res2=None):
    if gt2 is not None:
        hd = (hausdorff_distance(gt1, res1) + hausdorff_distance(gt2, res2)) / 2
    else:
        hd = hausdorff_distance(gt1, res1)
        
    return hd



def MAE(gt, res):
    
    return abs(gt - res)



def read2np(fname):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(fname))
    
    return image_array



def get_resolution(fname):
    
    return sitk.ReadImage(fname).GetSpacing()



if __name__ == "__main__":
    
    from load_data import load_data
    from step1_restoration import step1_restoration
    from step2_segmentation_lv import step2_segmentation_lv
    from step3_segmentation_myo import step3_segmentation_myo
    from step4_analysis import step4_analysis
    
    data_path = '../DataABO'
    
    fnames = glob.glob(data_path + '/**/*_4d_noisy.nii.gz')
    
    results = dict()
    results["step1_PSNR"] = []
    results["step1_SSIM"] = []
    
    results["step2_DICE_individual"] = []
    results["step2_Haussdorf_distance_individual"] = []
    
    results["step2_DICE_final"] = []
    results["step2_Haussdorf_distance_final"] = []
    
    results["step3_DICE_individual"] = []
    results["step3_Haussdorf_distance_individual"] = []
    
    results["step3_DICE_final"] = []
    results["step3_Haussdorf_distance_final"] = []
    
    results["step4_MAE_EF"] = []
    results["step4_MAE_MW"] = []
    
    for fnum, fname in enumerate(fnames):
        
        print(f"{fname}")
        with open(os.path.split(fname)[0] + '/Info.cfg') as f:
            info_cfg = yaml.safe_load(f)
        
        
        data_tzxy_np_restored_gt = read2np(fname.replace('4d_noisy.nii.gz', '4d.nii.gz'))
        ED_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '.nii.gz')
        ES_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '.nii.gz')
        ED_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz')
        ES_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz')
        ED_mask_lv_zxy_np_gt = ED_mask_zxy_np_gt == 3
        ES_mask_lv_zxy_np_gt = ES_mask_zxy_np_gt == 3
        ED_mask_myo_zxy_np_gt = ED_mask_zxy_np_gt == 2
        ED_resolution_gt = get_resolution(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz')
        
        ejection_fraction_gt = ((np.sum(ED_mask_lv_zxy_np_gt) - np.sum(ES_mask_lv_zxy_np_gt)) / np.sum(ED_mask_lv_zxy_np_gt)) * 100 # (%)
        
        rho = 1.05 / 1000 # average volumetric mass density of myocardium (g/mm^3)
        voxel_volume = ED_resolution_gt[0] * ED_resolution_gt[1] * ED_resolution_gt[2]  # mm^3
        myocard_weight_gt = np.sum(ED_mask_myo_zxy_np_gt) * voxel_volume * rho
        
        #TODO info vs. info_orig???
        
        data_tzxy_np, info_orig  = load_data(fname)
        
        data_tzxy_np_restored, info = step1_restoration(data_tzxy_np, info_orig)
        results["step1_PSNR"].append(PSNR(data_tzxy_np_restored_gt, data_tzxy_np_restored))
        results["step1_SSIM"].append(SSIM(data_tzxy_np_restored_gt, data_tzxy_np_restored))
        
        ED_data_zxy_np, ES_data_zxy_np, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, info = step2_segmentation_lv(data_tzxy_np_restored_gt, info_orig)
        results["step2_DICE_individual"].append(DICE(ED_mask_lv_zxy_np_gt, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np))
        results["step2_Haussdorf_distance_individual"].append(hausdorff(ED_mask_lv_zxy_np_gt, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np))
        
        ED_data_zxy_np, ES_data_zxy_np, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, info = step2_segmentation_lv(data_tzxy_np_restored, info_orig)
        results["step2_DICE_final"].append(DICE(ED_mask_lv_zxy_np_gt, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np))
        results["step2_Haussdorf_distance_final"].append(hausdorff(ED_mask_lv_zxy_np_gt, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np_gt, ES_mask_lv_zxy_np))
        
        ED_mask_myo_zxy_np, info = step3_segmentation_myo(ED_data_zxy_np_gt, ED_mask_lv_zxy_np_gt, info_orig)
        results["step3_DICE_individual"].append(DICE(ED_mask_myo_zxy_np_gt, ED_mask_myo_zxy_np))
        results["step3_Haussdorf_distance_individual"].append(hausdorff(ED_mask_myo_zxy_np_gt, ED_mask_myo_zxy_np))
        
        ED_mask_myo_zxy_np, info = step3_segmentation_myo(ED_data_zxy_np, ED_mask_lv_zxy_np, info_orig)
        results["step3_DICE_final"].append(DICE(ED_mask_myo_zxy_np_gt, ED_mask_myo_zxy_np))
        results["step3_Haussdorf_distance_final"].append(hausdorff(ED_mask_myo_zxy_np_gt, ED_mask_myo_zxy_np))
        
        ejection_fraction, myocard_weight = step4_analysis(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_myo_zxy_np, info_orig)
        results["step4_MAE_EF"].append(MAE(ejection_fraction_gt, ejection_fraction))
        results["step4_MAE_MW"].append(MAE(myocard_weight_gt, myocard_weight))
    
    
    
    results_mean = results.copy()
    for key in results_mean:
        results_mean[key] = np.round(np.mean(results_mean[key]), 4)
        
    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent = 6)
    
    with open("results_mean.json", "w") as json_file:
        json.dump(results_mean, json_file, indent = 6)
        
    data_frame_results_mean = pd.DataFrame(results_mean, index=[0]).T
    data_frame_results_mean.to_excel("results_mean.xlsx")
    
    print(data_frame_results_mean)
        