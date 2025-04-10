import torch
import torch.nn as nn
import torch.nn.functional as F

class REDiffusion(nn.Module):
    """
    The definition of Residual Estimation Diffusion(RED).
    REN: Residual Estimation Network
    DCN: Drift Correction Network
    Alphas and Betas are time schedule according to the paper
    """

    def __init__(self, nn_ren, nn_dcn, max_T, alphas, betas, device = "cuda"):
        super(REDiffusion, self).__init__()

        self.max_T = max_T
        self.device = device
        self.nn_ren = nn_ren.to(device)
        self.nn_dcn = nn_dcn.to(device)
        self.ratio = 1

        self.alphas = torch.linspace(0, 1, steps = max_T + 1).to(device) if alphas is None else alphas
        self.betas = torch.linspace(0, 1, steps = max_T + 1).to(device) if betas is None else betas

        self.training_step = 0

    def forward(self, x_t, t):
        "Predict the next state of xt, including residual estimation and drift correction."

        t_prev = t - self.ratio
        
        alpha_t = self.alphas[t]
        alpha_t_prev = self.alphas[t_prev]
        beta_t_prev = self.betas[t_prev]
        
        x_t_res = self.nn_ren(x_t,t / self.max_T)
        x_t_hat = x_t - (alpha_t - alpha_t_prev) * x_t_res
        x_t_drf = self.nn_dcn(x_t_hat, t_prev / self.max_T)
        x_t = x_t_hat - beta_t_prev * x_t_drf

        return x_t
    
    def save(self, save_path:str):
        "Save model and parameters"
        
        data = {
            'training_step' : self.training_step,
            'max_T'         : self.max_T,
            'alphas'        : self.alphas,
            'betas'         : self.betas,
            'nn_ren'        : self.nn_ren.state_dict(),
            'nn_dcn'        : self.nn_dcn.state_dict()
        }

        torch.save(data, str(f'{save_path}/model_{self.training_step}.pt'))

    def load(self, load_path:str):

        data = torch.load(load_path)

        self.training_step  = data["training_step"]
        self.max_T          = data["max_T"]
        self.alphas         = data["alphas"]
        self.betas         = data["betas"]
        self.nn_ren.load_state_dict(data['nn_ren'])
        self.nn_dcn.load_state_dict(data['nn_dcn'])

    def degrade(self, x_0, residual, t):
        "Degrade the input data according to the given residual and time step"
        
        b,c,h,w = x_0.shape
        x_t = x_0 + self.alphas[t].view((b, 1, 1, 1)) * residual

        return x_t

    def reverse(self, x_T, steps=50, need_corr = False):
        "The reverse process of RED"

        with torch.no_grad():

            init_time_step = int(self.max_T// 1)
            step_ratio = torch.tensor(init_time_step // steps).to(self.device)
            x_t = x_T
            b,c,h,w = x_T.shape

            t = torch.tensor(self.max_T).tile(b).to(self.device)
            t_prev = t - step_ratio

            for i in range(steps, 0, -1):

                alpha_t = self.alphas[t].reshape(-1,1,1,1)
                alpha_t_prev = self.alphas[t_prev].reshape(-1,1,1,1)
                beta_t_prev = self.betas[t_prev].reshape(-1,1,1,1)

                pred_residual = self.nn_ren(x_t, t)
                x_t_hat = x_t - (alpha_t - alpha_t_prev) * pred_residual
                if(need_corr):
                    pred_drift = self.nn_dcn(x_t_hat, t)
                    x_t = x_t_hat + beta_t_prev * pred_drift
                else:
                    x_t = x_t_hat

                t = t - step_ratio
                t_prev = t - step_ratio

            return x_t

def get_diffusion(max_timestep = 1000, path = None, device = "cuda"):
    from nnet.Model_Unet import get_unet

    timesteps = torch.linspace(1, 0, steps=max_timestep + 1).to(device)
    alphas = (1 - timesteps ** 2).clamp(min=0.0001, max=1.0)
    betas = (1 - timesteps) ** 3
    betas /= betas.sum()
    
    diffusion = REDiffusion(
        nn_ren = get_unet(),
        nn_dcn = get_unet(),
        max_T = max_timestep,
        alphas = alphas,
        betas = betas,
        device = device
    )

    if path is not None:
        diffusion.load(path)

    return diffusion

