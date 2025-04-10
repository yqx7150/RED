
import json
import torch
import numpy as np
import torch.nn.functional as F

from math import exp
from torch.autograd import Variable

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def load_config(config_path = "config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def MSELoss(input, target):
    loss = F.mse_loss(input, target)
    return loss

def SmoothL1Loss(input, target):
    loss = F.smooth_l1_loss(input,target)
    return loss

def SSIMLoss(img1, img2, window_size=11, channel=1, size_average=True):

    sigma = 1.5
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    window_1D = (gauss/gauss.sum()).unsqueeze(1)
    window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window_2D.expand(channel, 1, window_size, window_size).contiguous()).to(img1.device)
    
    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_loss = 1 - ssim_map.mean()
    return ssim_loss

def batch_ssim(img_1: torch.Tensor, img_2: torch.Tensor):

    batch_size = img_1.size(0)
    ssim_values = []
    
    for i in range(batch_size):
        img1_np = img_1[i][0].detach().cpu().numpy()
        img2_np = img_2[i][0].detach().cpu().numpy()
        
        ssim_value = ssim(img1_np+ 1e-10, img2_np+ 1e-10, data_range=img2_np.max() + 1e-10)

        ssim_values.append(ssim_value)
    
    return np.mean(ssim_values)

def batch_psnr(img_1: torch.Tensor, img_2: torch.Tensor, max_value = None):

    batch_size = img_1.size(0)
    psnr_values = []
    
    for i in range(batch_size):
        img1_np = img_1[i][0].detach().cpu().numpy()
        img2_np = img_2[i][0].detach().cpu().numpy()
        
        max_value = img2_np.max() if max_value is None else max_value
        psnr_value = np.clip(psnr(img1_np, img2_np, data_range=max_value),0,100)
        psnr_value = 0 if np.isnan(psnr_value) else psnr_value
        
        psnr_values.append(psnr_value)
    
    return np.mean(psnr_values)

def batch_nrmse(img_1: torch.Tensor, img_2: torch.Tensor):

    batch_size = img_1.size(0)
    nrmse_values = []
    
    for i in range(batch_size):
        img1_np = img_1[i][0].detach().cpu().numpy()
        img2_np = img_2[i][0].detach().cpu().numpy()
        
        nrmse_value =  np.clip(nrmse(img1_np, img2_np, normalization='min-max'),0,10)
        nrmse_value = 10 if np.isnan(nrmse_value) else nrmse_value

        nrmse_values.append(nrmse_value)
    
    return np.mean(nrmse_values)


