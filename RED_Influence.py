
import os
import torch
import pydicom
import numpy as np

from PIL import Image

from nnet.Model_Diffusion import get_diffusion
from RED_Projection import pet_to_sino, sino_to_pet, set_projection_size
from nnet.Model_Utility import batch_psnr, batch_ssim, batch_nrmse

model_path = "path to trained dcn model"
# model_path = "./models/DCN_D20_20250412_205402/model_580000.pt"

device = "cuda:0"
set_projection_size((360,360))

def load_dcm(dicom_file):
    return pydicom.dcmread(dicom_file)
    
def calculate_suv(pixel_array, ds):
    
    image_data = pixel_array
    patient_weight = ds.PatientWeight
    radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
    injected_dose = radiopharmaceutical_info.RadionuclideTotalDose
    activity_concentration = image_data
    suv = (activity_concentration * patient_weight * 1000) / injected_dose

    return suv

def influence_RED(low_dose_sino_path = "", norm_dose_sino_path = ""):
    with torch.no_grad():

        diffusion = get_diffusion(path = model_path, device = device)

        low_dcm = load_dcm(low_dose_sino_path)
        norm_dcm = load_dcm(norm_dose_sino_path)

        low_sino = pet_to_sino(low_dcm.pixel_array)
        norm_sino = pet_to_sino(norm_dcm.pixel_array)

        low_sino = calculate_suv(low_sino,low_dcm)
        norm_sino = calculate_suv(norm_sino,norm_dcm)

        max_v =low_sino.max()
        low_dose_sino =  low_sino / max_v
        norm_dose_sino = norm_sino / max_v

        low_sino_tensor = torch.tensor(low_dose_sino)[None,None,:].to(device)
        norm_sino_tensor = torch.tensor(norm_dose_sino)[None,None,:].to(device)

        x_T = (low_sino_tensor - 0.5 ) * 2
        pred_sino = diffusion.reverse(x_T, 50, need_corr= True)

        pred_sino = pred_sino[0][0].detach().to("cpu").numpy() / 2 + 0.5
        pred_img = sino_to_pet(pred_sino)

        from PIL import Image
        normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        Image.fromarray(255 * normalize(pred_sino)).show()
        Image.fromarray(255 * normalize(pred_img)).show()


    return pred_sino

    
if __name__ == '__main__':
    ld_path = r"path to low dows dcm"
    norm_path = r"path to norm dows dcm"

    rec_img = influence_RED(ld_path,norm_path)

    print("done")