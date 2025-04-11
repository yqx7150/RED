
import os
import torch
import pydicom
import numpy as np

from PIL import Image

from nnet.Model_Diffusion import get_diffusion
from RED_Projection import pet_to_sino, set_projection_size

model_path = "path to trained DCN.pt"
set_projection_size(360,360)


def load_dcm(dicom_file):
    return pydicom.dcmread(dicom_file)
    
def calculate_suv(ds):
    
    image_data = ds.pixel_array
    patient_weight = ds.PatientWeight  # 例如: 70.0 kg
    radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
    injected_dose = radiopharmaceutical_info.RadionuclideTotalDose  # 例如 50000000 Bq
    activity_concentration = image_data  # 例如: 3.5 Bq/mL
    suv = (activity_concentration * patient_weight * 1000) / injected_dose

    return suv

def influence_RED(low_dose_sino_path = "", norm_dose_sino_path = ""):
    global test_params
    with torch.no_grad():

        diffusion = get_diffusion(path = test_params["model_path"]).to("cuda")

        alpha = diffusion.alphas.to("cuda")

        low_dcm = load_dcm(low_dose_sino_path)
        norm_dcm = load_dcm(norm_dose_sino_path)

        img = np.clip(calculate_suv(low_dcm),0,10)
        target = np.clip(calculate_suv(norm_dcm),0,10)
        low_sino = pet_to_sino(img)
        norm_sino = pet_to_sino(target)

        low_sino = np.array(Image.fromarray(low_sino).resize((352,352)))
        norm_sino = np.array(Image.fromarray(norm_sino).resize((352,352)))
        
        low_sino_tensor = torch.tensor(low_sino)[None,None,:]
        norm_sino_tensor = torch.tensor(norm_sino)[None,None,:]

        max_v = low_sino_tensor.max()
        val_low_dose_sino =  (max_v * low_sino_tensor / torch.clamp(max_v**2,min=1)).to("cuda")
        val_norm_dose_sino = (max_v * norm_sino_tensor / torch.clamp(max_v**2,min=1)).to("cuda")

        pred_sino = diffusion.reverse(val_low_dose_sino, 50, need_corr= True)

    return pred_sino

    
if __name__ == '__main__':
    ld_path = r"E:\dataset\uExplorer\PART3\PART3\Anonymous_ANO_20230505_1758162_120417\2.886 x 600 WB D10\00000251.dcm"
    norm_path = r"E:\dataset\uExplorer\PART3\PART3\Anonymous_ANO_20230505_1758162_120417\2.886 x 600 WB NORMAL\00000251.dcm"
    rec_img = influence_RED(ld_path,norm_path)

    print("done")