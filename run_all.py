
from load_data import load_data
from step1_restoration import step1_restoration
from step2_segmentation_lv import step2_segmentation_lv
from step3_segmentation_myo import step3_segmentation_myo
from step4_analysis import step4_analysis



if __name__ == "__main__":
    
    
    data_tzxy_np, info  = load_data('../DataABO/patient061/patient061_4d.nii.gz')
    
    data_tzxy_np_restored, info = step1_restoration(data_tzxy_np, info)
    
    ED_data_zxy_np, ES_data_zxy_np, ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, info = step2_segmentation_lv(data_tzxy_np_restored, info)
    
    ED_mask_myo_zxy_np, info = step3_segmentation_myo(ED_data_zxy_np, ED_mask_lv_zxy_np, info)
    
    ejection_fraction, myocard_weight = step4_analysis(ED_mask_lv_zxy_np, ES_mask_lv_zxy_np, ED_mask_myo_zxy_np, info)
    
    print(f"Ejection fraction = {ejection_fraction} %\nMyocard weight = {myocard_weight} g")
    