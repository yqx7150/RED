# RED
**Paper:** Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction
**Authors**: Xingyu Ai, Bin Huang, Fang Chen, Liu Shi, Binxuan Li, Shaoyu Wang*, Qiegen Liu*

Published on Medical Image Analysis(2025), https://www.sciencedirect.com/science/article/pii/S1361841525001057

Date : April-10-2025 
Version : 1.0
The code and the algorithm are for non-comercial use only.
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.


Recent advances in diffusion models have demonstrated exceptional performance in generative tasks across various fields. In positron emission tomography (PET), the reduction in tracer dose leads to information loss in sinograms. Using diffusion models to reconstruct missing information can improve imaging quality. Traditional diffusion models effectively use Gaussian noise for image reconstructions. However, in low-dose PET recon-struction, Gaussian noise can worsen the already sparse data by introducing artifacts and inconsistencies. To address this issue, we propose a diffusion model named residual estimation diffusion (RED). From the perspec-tive of diffusion mechanism, RED uses the residual between sinograms to replace Gaussian noise in diffusion process, respectively sets the low-dose and full-dose sinograms as the starting point and endpoint of reconstruc-tion. This mechanism helps preserve the original information in the low-dose sinogram, thereby enhancing re-construction reliability. From the perspective of data consistency, RED introduces a drift correction strategy to reduce accumulated prediction errors during the reverse process. Calibrating the intermediate results of reverse iterations helps maintain the data consistency and enhances the stability of reconstruction process. In the ex-periments, RED achieved the best performance across all metrics. 

![Main](/images/Fig3.png)

The above figure demonstrated the overall structure of RED. During the reverse process, REN first estimates the residual, then DCN predicts the drift correction term. The experiments were conducted on a Windows 10 operating system. The following sections will demonstrate how to deploy and use the Residual Estimation Diffusion (RED) method.

## Data Preparation
The model is trained in a two-stage manner. First, the data needs to be organized and packaged into a training dataset. Users can modify the process using the reference code in Tool_DataPrepare.py.

## Training
The model utilizes a two-stage training approach. In the first stage, REN is trained. After modifying the config_ren.json file, the following command should be executed to initiate the training.

```bash
python RED_Training_REN.py
```
After the training, the appropriate model can be selected based on the training results. Then, the config_dcn.json file should be modified, followed by executing the command below to train the DCN.

```bash
python RED_Training_DCN.py
```

## Experiment Result

![Main](/images/Fig4.png)

(Content under editing...)