# RED
**Paper:** Residual Estimation Diffusion for Low-Dose PET Sinogram Reconstruction
**Authors**: Xingyu Ai, Bin Huang, Fang Chen, Liu Shi, Binxuan Li, Shaoyu Wang*, Qiegen Liu*

Published on Medical Image Analysis(2025), https://www.sciencedirect.com/science/article/pii/S1361841525001057

Date : April-10-2025 
Version : 1.0
The code and the algorithm are for non-comercial use only.
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.


Recent advances in diffusion models have demonstrated exceptional performance in generative tasks across various fields. In positron emission tomography (PET), the reduction in tracer dose leads to information loss in sinograms. Using diffusion models to reconstruct missing information can improve imaging quality. Traditional diffusion models effectively use Gaussian noise for image reconstructions. However, in low-dose PET recon-struction, Gaussian noise can worsen the already sparse data by introducing artifacts and inconsistencies. To address this issue, we propose a diffusion model named residual estimation diffusion (RED). From the perspec-tive of diffusion mechanism, RED uses the residual between sinograms to replace Gaussian noise in diffusion process, respectively sets the low-dose and full-dose sinograms as the starting point and endpoint of reconstruc-tion. This mechanism helps preserve the original information in the low-dose sinogram, thereby enhancing re-construction reliability. From the perspective of data consistency, RED introduces a drift correction strategy to reduce accumulated prediction errors during the reverse process. Calibrating the intermediate results of reverse iterations helps maintain the data consistency and enhances the stability of reconstruction process. In the ex-periments, RED achieved the best performance across all metrics. 

![Fig1](/images/Fig1.png)
The fundamental concept of RED is to perform reconstruction by estimating the residual between low-dose and full-dose sinogram. In this framework, the low-dose image is treated as a noise-corrupted version of the full-dose sinogram, which serves as the foundation for constructing the diffusion process.

![Fig2](/images/Fig2.png)
![Fig3](/images/Fig3.png)

The above figure demonstrated the overall structure of RED. During the reverse process, REN first estimates the residual, then DCN predicts the drift correction term. The experiments were conducted on a Windows operating system. 

The following sections will demonstrate how to deploy and use this project.

## Data Preparation
The model is trained in a two-stage manner. First, the data needs to be organized and packaged into a training dataset. Users can modify the process using the reference code in Tool_DataPrepare.py.

## Training
The model utilizes a two-stage training approach. In the first stage, REN is trained. After modifying the config_ren.json file, the following command should be executed to initiate the training.

```bash
python RED_Training_REN.py
```
After training REN, the appropriate model can be selected based on the training results. Then, the config_dcn.json file should be modified, followed by executing the command below to train the DCN.

```bash
python RED_Training_DCN.py
```

The performance can be evaluated by running the provided influence script, with the DICOM image format specified within the code.

```bash
python RED_Influence.py
```

## Experiment Result
To validate the performance of RED, this section compares it with OSEM (Hudson and Larkin, 1994), U-Net (Ronneberger et al., 2015), DDIM (Song et al., 2020), cold diffusion (CD) (Bansal et al., 2023) and denoising diffusion bridge model (DDBM) (Zhou et al., 2023) in reconstruction tasks. 
### Reconstruction Results Under DRF 4
![Fig4](/images/Fig4.png)
### Reconstruction Results Under DRF 20
![Fig5](/images/Fig5.png)
### Reconstruction Results Under DRF 100
![Fig6](/images/Fig6.png)
### Table
![Table](/images/Table1.png)

Due to the high quality of the initial data, all models performrelatively well under DRF 4, with RED achieving the highest PSNRvalue of 39.57 dB. Furthermore, RED consistently delivered the bestresults under DRF 20. This trend became even more pronounced at DRF 100, where RED outperforms OSEM by 8.08 dB and surpasses the second-ranked CD by 2.23 dB in terms of PSNR. Figs.4-6 display the sinograms and reconstruction results at different dose levels.Overall, the results of RED are smoother while retaining more details. At DRF4, images of RED exhibit fewer noise and artifacts, whereas othermethods signify more graininess within the organs. Under DRF 20, thedata quality degraded, and the unsupervised diffusion methods perform poorly. Although U-Net manages to reconstruct the lostinformation, it produces blurry images. In contrast, results of RED are moresimilar to the full-dose images, achieving smooth interiors whilepreserving sharp edges. At DRF 100, the result of OSEM becomes barely recognizable, andU- Net introduces considerable blurring. Both DDIM and CD recoverfew details. The performance of DDBM degrades noticeably in thisscenario, with the reconstructed results showing over-smoothing, alongwith an increased presence of artifacts and distortions. Despite theextremely poor conditions, RED still manages to restore part of theinformation to maintain structural integrity. 