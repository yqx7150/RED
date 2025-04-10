"""
Prepair dataset for training, this code will scan all .npy files in a given path (includ subdirectories).
You may need to modify these process for your own dataset according to the definition of dataset.
"""

import os
import numpy as np
from tqdm import tqdm
from RED_Projection import pet_to_sino,calculate_suv,load_dcm,sino_to_pet,set_projection_size

normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)

def idx_to_dirIdx(idx, p, d, c):
    idx_0 = idx
    idx_1 = idx_0 // c
    idx_2 = 0
    idx_3 = idx_0 - idx_1 * c
    return (idx_1,0,idx_3)

def get_folder_files_path(main_folder,file_type):
    "get all files names in a given folder"

    subdirs = []
    files = [file for file in os.listdir(main_folder) if file.endswith(file_type)]
    
    for file in tqdm(files, desc="Processing"):
        file_path = os.path.join(main_folder, file)
        subdirs.append(file_path)
    
    return np.array(subdirs, dtype=object)
    
def get_all_datas_path(main_folder,file_type):
    "Scan and return all files of the given type"

    all_paths = []

    subdir1_list = [os.path.join(main_folder, subdir1) for subdir1 in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subdir1))]
    
    for subdir1_path in tqdm(subdir1_list, desc="Processing subdir1"):
        subdir1_paths = []
        subdir2_list = [os.path.join(subdir1_path, subdir2) for subdir2 in os.listdir(subdir1_path) if os.path.isdir(os.path.join(subdir1_path, subdir2))]
        # print(str(len(subdir1_path)))
        for subdir2_path in subdir2_list:
            subdir2_paths = []
            dcm_files = [file for file in os.listdir(subdir2_path) if file.endswith(file_type)]
            
            for file in dcm_files:
                file_path = os.path.join(subdir2_path, file)
                subdir2_paths.append(file_path)
            
            subdir1_paths.append(subdir2_paths)
        
        all_paths.append(subdir1_paths)

    all_paths_np = np.array(all_paths, dtype=object)

    return all_paths_np

def generate_dataset(
        data_paths, save_img_path = None, save_sino_path = None, 
        img_size=(360, 360), 
        sino_size=(360, 360), 
        start_idx=0, max_num=100000, need_SUV = False):

    p,d,c = data_paths.shape
    total_count = p * 1 * c

    if save_img_path != None and not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if save_sino_path != None and not os.path.exists(save_sino_path):
        os.makedirs(save_sino_path)
    
    set_projection_size(sino_size)
    
    for idx in tqdm(range(total_count)):

        idx_1,idx_2,idx_3 = idx_to_dirIdx(idx,p,d,c)

        low_dcm_path = data_paths[idx_1,idx_2,idx_3]
        norm_dcm_path = data_paths[idx_1,-1,idx_3]
        low_dcm = load_dcm(low_dcm_path)
        norm_dcm = load_dcm(norm_dcm_path)

        if need_SUV:
            low_img_data = calculate_suv(low_dcm)
            norm_img_data = calculate_suv(norm_dcm)
        else:
            low_img_data = low_dcm.pixel_array
            norm_img_data = norm_dcm.pixel_array

        dcm_path_pair = [low_dcm_path,norm_dcm_path]

        if save_img_path is not None:
            img_datapair= np.zeros((2, img_size[0], img_size[1]), dtype=np.float32)
            img_datapair[0] = low_img_data
            img_datapair[1] = norm_img_data
            np.savez(f"{save_img_path}\\img_{idx + start_idx}", img = img_datapair,dcm_path = dcm_path_pair)
            # np.savez_compressed(f"./datasets/comp\\img_{idx + start_idx}", img = img_datapair,dcm_path = dcm_path_pair)

        if save_sino_path is not None:
            sino_datapair= np.zeros((2, img_size[0], img_size[1]), dtype=np.float32)
            sino_datapair[0] = pet_to_sino(low_img_data)
            sino_datapair[1] = pet_to_sino(norm_img_data)
            np.savez(f"{save_sino_path}\\sino_{idx + start_idx}", allow_pickle=True, sino_image=sino_datapair, dcm_path = dcm_path_pair)
            # np.savez_compressed(f"./datasets/comp\\sino_{idx + start_idx}",allow_pickle=True, sino_image=sino_datapair, dcm_path = dcm_path_pair)

if __name__ == '__main__':

    # 1. get all files' path with the given format
    all_img_paths = get_all_datas_path( r'E:\dataset\uExplorer\PART1\PART1',file_type=".dcm")
    np.save('./datasets/p1_d10_paths.npy', all_img_paths[:,[0,5],:])
    np.save('./datasets/p1_d100_paths.npy', all_img_paths[:, [1,5], :])
    np.save('./datasets/p1_d20_paths.npy', all_img_paths[:, [2,5], :])
    np.save('./datasets/p1_d4_paths.npy', all_img_paths[:, [3,5], :])
    np.save('./datasets/p1_d50_paths.npy', all_img_paths[:, [4,5], :])
    np.save('./datasets/p1_all_paths.npy', all_img_paths[:, :, :])

    data_name = "p1_d20_paths.npy"
    paths = np.load(f'./datasets/{data_name}', allow_pickle=True)
    print(f"Start{data_name}")

    # 2. prepare sinograms and save them for training
    generate_dataset(
        paths,
        save_sino_path = "./datasets/Sino_D20",
        save_img_path = None,
        img_size=(360,360),
        start_idx=0,
        max_num=1000000
    )

    # 3. pack all paths as one file
    all_sino_paths = get_folder_files_path("./datasets/Sino_D20",".npz")
    np.save("./datasets/d20_sino_paths", all_sino_paths)

    print("done!")