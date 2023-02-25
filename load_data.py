import napari
import SimpleITK as sitk
import numpy as np

def load_data(fname):
    '''
    This is function for loading of Nifty data to numpy array and metadata tu custom format (info).
    
    Parameters
    ----------
    fname : str
        filename - complete path to .nii.gz file.

    Returns
    -------
    data_tzxy_np : numpy array with np.float64 dtype, axis order tzxy
        4D MRI data of heard.
    info : any data type
        custom set of required values for other steps and metadata.
    '''
    
    
    data_tzxy_np = sitk.GetArrayFromImage(sitk.ReadImage(fname)).astype(np.float32)
    
    info = ['can contain any custom data', 42]
    
    return data_tzxy_np, info



if __name__ == "__main__":
    
    
    data_tzxy_np, info  = load_data('../DataABO/patient061/patient061_4d.nii.gz')
    
    print(info)
    viewer = napari.view_image(data_tzxy_np)
    napari.run()