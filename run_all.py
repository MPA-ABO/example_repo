
from load_data import load_data
from step1_restoration import step1_restoration
from step2_segmentation import step2_segmentation
from step3_anylysis import step3_anylysis



if __name__ == "__main__":
    
    
    data_tzxy_np, info  = load_data('../DataABO/patient061/patient061_4d.nii.gz')
    
    data_tzxy_np_restored, info = step1_restoration(data_tzxy_np, info)
    
    ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info = step2_segmentation(data_tzxy_np_restored, info)
    
    ejection_fraction, myocard_volume = step3_anylysis(ED_data_zxy_np, ES_data_zxy_np, ED_mask_zxy_np, ES_mask_zxy_np, info)
    
    