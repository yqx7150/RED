
import pydicom
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset, ConcatDataset

from RED_Projection import set_projection_size

class Sino_DCM_Dataset(Dataset):
    
    def __init__(self, npy_file):
        self.all_sino_np = np.load(npy_file, allow_pickle=True)
        self.total_len = len(self.all_sino_np)
        
    def __len__(self):
        return self.total_len

    def cal_suv(self, data, ds):

        # Weight (kg)
        patient_weight = ds.PatientWeight
        
        # Dose (MBq)
        radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = radiopharmaceutical_info.RadionuclideTotalDose
        
        activity_concentration = data

        suv = (activity_concentration * patient_weight * 1000) / injected_dose
        
        return suv

    def __getitem__(self, idx):

        cur_data = np.load(self.all_sino_np[idx])

        low_dose_sino   = cur_data["sino_image"][0] # low dose sinograme
        norm_dose_sino  = cur_data["sino_image"][1] # full dose sinograme
        ld_dcm_path     = cur_data["dcm_path"][0] # low dose dcm path
        nd_dcm_path     = cur_data["dcm_path"][1] # full dose dcm path

        low_dose_sino = self.cal_suv(low_dose_sino,pydicom.dcmread(ld_dcm_path))
        norm_dose_sino = self.cal_suv(norm_dose_sino,pydicom.dcmread(nd_dcm_path))

        return [
            low_dose_sino[None,:],
            norm_dose_sino[None,:],
            str(ld_dcm_path),
            str(nd_dcm_path)
        ]



def create_dataloader(params):

    datasets_path = params["datasets_path"]
    batch_size = params["batch_size"]
    sino_size = params["sino_size"]

    total_dataset = []
    for dataset_path in datasets_path:
        total_dataset.append(Sino_DCM_Dataset(dataset_path))
    total_dataset = ConcatDataset(total_dataset)
    
    set_projection_size(sino_size)

    train_dataset = Subset(total_dataset, range(0,int(len(total_dataset) * 0.95)))
    val_dataset = Subset(total_dataset, range(int(len(total_dataset) * 0.95), len(total_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory = True, num_workers=0)
    val_dataloader = iter(DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    return train_dataloader, val_dataloader
    