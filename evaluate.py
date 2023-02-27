import glob
import SimpleITK as sitk
import numpy as np
import yaml
import os
import json

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from skimage.metrics import hausdorff_distance



def PSNR(image_true, image_test):
    return peak_signal_noise_ratio(image_true, image_test, data_range=1000)
  
def SSIM(image_true, image_test):
    
    # 2D?
    # ssims = []
    # for t in range(image_true.shape[0]):
    #     ssim = structural_similarity(image_true[0, ...], image_test[0, ...], data_range=1000, channel_axis=0)
    #     ssims.append(ssim)
    # 
    # return np.mean(ssims)
    
    return structural_similarity(image_true, image_test, data_range=1000, channel_axis=0)
  

def DICE(gt1, gt2, res1, res2):
    
    gt = (np.stack((gt1, gt2)) > 0).astype(np.float32)
    seg = (np.stack((res1, res2)) > 0).astype(np.float32)
    
    return np.sum(seg[gt == True])*2.0 / (np.sum(seg) + np.sum(gt))




# def hausdorff_distance(gt, res):

#     hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    
#     hausdorff_distance_filter.Execute(sitk.GetImageFromArray(gt.astype(np.int16),isVector=False), sitk.GetImageFromArray(res.astype(np.int16),isVector=False))
    
#     return hausdorff_distance_filter.GetHausdorffDistance()
    
    

def hausdorff(gt1, gt2, res1, res2):
    
    # return (hausdorff_distance(gt1, res1) + hausdorff_distance(gt2, res2)) / 2
    
    
    hausdorffs = []
    for z in range(gt1.shape[0]):
        
        hausdorffs.append(hausdorff_distance(gt1[z, ...], res1[z, ...]))
        hausdorffs.append(hausdorff_distance(gt2[z, ...], res2[z, ...]))
    
    return np.mean(hausdorffs)



def read2np(fname):
    
    return sitk.GetArrayFromImage(sitk.ReadImage(fname))


if __name__ == "__main__":
    
    from load_data import load_data
    from step1_restoration import step1_restoration
    from step2_segmentation import step2_segmentation
    from step3_analysis import step3_analysis
    
    data_path = '../DataABO'
    
    fnames = glob.glob(data_path + '/**/*_4d_noisy.nii.gz')
    
    results = dict()
    results["step1_PSNR"] = []
    results["step1_SSIM"] = []
    
    results["step2_DICE_individual"] = []
    results["step2_Haussdorf_distance_individual"] = []
    
    results["step2_DICE_final"] = []
    results["step2_Haussdorf_distance_final"] = []
    
    results["step3_EF_MSE_individual"] = []
    results["step3_MW_MSE_individual"] = []
    
    results["step3_EF_MSE_final"] = []
    results["step3_MW_MSE_final"] = []
    
    for fnum, fname in enumerate(fnames):
        
        
        with open(os.path.split(fname)[0] + '/Info.cfg') as f:
            info_cfg = yaml.safe_load(f)
        
        
        data_tzxy_np_restored_gt = read2np(fname.replace('4d_noisy.nii.gz', '4d.nii.gz'))
        ED_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '.nii.gz')
        ES_data_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '.nii.gz')
        ED_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ED']).zfill(2) + '_gt.nii.gz')
        ES_mask_zxy_np_gt = read2np(fname.replace('4d_noisy.nii.gz','') + 'frame' + str(info_cfg['ES']).zfill(2) + '_gt.nii.gz')
        ejection_fraction_gt = (np.sum(ED_mask_zxy_np_gt == 3) - np.sum(ES_mask_zxy_np_gt == 3)) / np.sum(ED_mask_zxy_np_gt == 3)
        myocard_weight_gt = info_cfg['Weight']
        

        data_tzxy_np, info_orig  = load_data(fname)
        
        
        data_tzxy_np_restored, info = step1_restoration(data_tzxy_np, info_orig)
        results["step1_PSNR"].append(PSNR(data_tzxy_np_restored_gt, data_tzxy_np_restored))
        results["step1_SSIM"].append(SSIM(data_tzxy_np_restored_gt, data_tzxy_np_restored))
        
        ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info = step2_segmentation(data_tzxy_np_restored_gt, info_orig)
        results["step2_DICE_individual"].append(DICE(ED_mask_zxy_np, ES_mask_zxy_np, ED_mask_zxy_np_gt, ES_mask_zxy_np_gt))
        
        ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info = step2_segmentation(data_tzxy_np_restored_gt, info_orig)
        results["step2_DICE_final"].append(DICE(ED_mask_zxy_np, ES_mask_zxy_np, ED_mask_zxy_np_gt, ES_mask_zxy_np_gt))
        
        
        # ejection_fraction, myocard_volume = step3_anylysis(ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info)
        
        
        
    results_mean = results.copy()
    for key in results_mean:
        results_mean[key] = np.mean(results_mean[key])
        
    print(results_mean)
        
        
    with open('results_mean.json', 'w') as json_file:
        json.dump(results_mean, json_file, indent = 6)
    
    with open('results.json', 'w') as json_file:
        json.dump(results, json_file, indent = 6)