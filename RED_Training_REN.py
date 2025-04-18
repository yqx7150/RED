
import torch

from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW

from RED_Projection import sino_to_pet
from RED_Dataset import create_dataloader

from nnet.Model_Diffusion import get_diffusion
from nnet.Model_Utility import MSELoss, SSIMLoss, normalize, batch_psnr, batch_ssim, batch_nrmse, load_config

from torch.utils.tensorboard import SummaryWriter

def train_ren(params):

    batch_size = params["batch_size"]
    epoch_num = params["epoch_num"]
    lr = params["lr"]
    lamb = params["lambda"]
    max_timestep = params["max_timestep"]
    device = params["device"]

    train_dataloader, val_dataloader = create_dataloader(params)

    diffusion = get_diffusion(max_timestep = max_timestep, path = params["pre_trained_diffusion"], device = device).to(device)

    optimizer = AdamW([
        {'params': diffusion.nn_ren.parameters(), 'lr': lr}
        # {'params': diffusion.nn_dcn.parameters(), 'lr': lr}
    ])

    alphas = diffusion.alphas.to(device)
    betas = diffusion.betas.to(device)
    global_step = 0

    # Create tensorboard writer
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{params['save_path_root']}_{current_time}"
    print(f"Start {save_path}")
    writer = SummaryWriter(save_path)

    for epoch in range(epoch_num):
        progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{epoch_num}", total=len(train_dataloader))

        for batch_idx, batch in progress_bar:

            optimizer.zero_grad()

            max_v = torch.clamp(batch[0].view(batch_size, -1).max(dim=1, keepdim=True)[0][:, None, None], min=0.1).to(device)
            low_dose_sino = (batch[0].to(device) / max_v - 0.5 ) * 2
            norm_dose_sino = (batch[1].to(device) / max_v - 0.5 ) * 2

            ts = torch.randint(1, max_timestep + 1, (batch_size,)).to(device)
            alpha_t = alphas[ts].reshape(-1,1,1,1)
            beta_t = betas[ts].reshape(-1,1,1,1)

            x_T = low_dose_sino
            x_0 = norm_dose_sino
            residual = x_T - x_0

            x_t = diffusion.degrade(x_0,residual,ts)

            pred_residual = diffusion.nn_ren(x_t, ts)

            pred_x_0 = x_t - alpha_t * pred_residual
            residual_loss = MSELoss(pred_residual, residual)
            pred_loss = lamb * SSIMLoss(pred_x_0, x_0)
            total_loss = residual_loss + pred_loss
            total_loss.backward()

            optimizer.step()

            writer.add_scalar(f'Train/Loss_1_Residual', residual_loss.detach().cpu().item(), global_step)
            writer.add_scalar(f'Train/Loss_3_PredSim', pred_loss.detach().cpu().item(), global_step)
            writer.add_scalar(f'Train/Total Loss', total_loss.item(), global_step)

            progress_bar.set_postfix({
                "Total Loss": total_loss.item(),
            })

            need_evl = False

            if need_evl or global_step % 1000 == 0:
                
                with torch.no_grad():
                    val_batch = next(val_dataloader)

                    val_max_v = torch.clamp(val_batch[0].view(batch_size, -1).max(dim=1, keepdim=True)[0][:, None, None], min=0.1).to(device)
                    val_low_dose_sino = val_batch[0].to(device) / val_max_v
                    val_norm_dose_sino = val_batch[1].to(device) / val_max_v

                    val_x_T = (val_low_dose_sino - 0.5 ) * 2
                    val_x_0 = val_norm_dose_sino
                    val_x_t = diffusion.reverse(val_x_T, 50, False)
                    val_x_t = val_x_t / 2 + 0.5

                    val_inp_img_np = (val_x_T[0][0]).detach().to("cpu").numpy()
                    writer.add_image(f'Images/1 Low Dose', normalize(val_inp_img_np), global_step, dataformats='HW')
                    norm_dose_img_np = ((val_norm_dose_sino)[0][0]).detach().to("cpu").numpy()
                    writer.add_image(f'Images/2 Normal Dose', normalize(norm_dose_img_np), global_step, dataformats='HW')
                    pred_sino_np =  (val_x_t[0][0]).detach().to("cpu").numpy()
                    writer.add_image(f'Images/3 Predicted', normalize(pred_sino_np), global_step, dataformats='HW')

                    psnr = batch_psnr(val_x_t, val_norm_dose_sino)
                    writer.add_scalar(f'Measure/PSNR', psnr.item(), global_step)
                    ssim = batch_ssim(val_x_t, val_norm_dose_sino)
                    writer.add_scalar(f'Measure/SSIM', ssim.item(), global_step)
                    nrmse = batch_nrmse(val_x_t, val_norm_dose_sino)
                    writer.add_scalar(f'Measure/NRMSE', nrmse.item(), global_step)

                    avg_pet_psnr = 0
                    avg_pet_ssim = 0
                    avg_pet_nrmse = 0

                    for i in range(batch_size):
                    
                        val_pred_pet_np = sino_to_pet((val_x_t)[0][0].detach().to("cpu").numpy())
                        val_norm_pet_np = sino_to_pet((val_norm_dose_sino)[0][0].detach().to("cpu").numpy())
                        val_low_pet_np =  sino_to_pet((val_low_dose_sino)[0][0].detach().to("cpu").numpy())

                        reversed_pred_pet = torch.tensor(val_pred_pet_np)[None,None]
                        reversed_norm_pet = torch.tensor(val_norm_pet_np)[None,None]
                        reversed_low_pet = torch.tensor(val_low_pet_np)[None,None]

                        psnr_pet = batch_psnr(reversed_pred_pet, reversed_norm_pet)
                        ssim_pet = batch_ssim(reversed_pred_pet, reversed_norm_pet)
                        nrmse_pet = batch_nrmse(reversed_pred_pet, reversed_norm_pet)
                        
                        avg_pet_psnr += psnr_pet/batch_size
                        avg_pet_ssim += ssim_pet/batch_size
                        avg_pet_nrmse += nrmse_pet/batch_size

                    writer.add_image(f'Images/5 Norm_PET', normalize(val_norm_pet_np), global_step, dataformats='HW')
                    writer.add_image(f'Images/4 Low_PET', normalize(val_low_pet_np), global_step, dataformats='HW')
                    writer.add_image(f'Images/6 Predicted_PET', normalize(val_pred_pet_np), global_step, dataformats='HW')

                    writer.add_scalar(f'Measure/PSNR_PET', avg_pet_psnr.item(), global_step)
                    writer.add_scalar(f'Measure/SSIM_PET', avg_pet_ssim.item(), global_step)
                    writer.add_scalar(f'Measure/NRMSE_PET', avg_pet_nrmse.item(), global_step)

                    del val_max_v, val_low_dose_sino, val_norm_dose_sino, val_x_T, val_x_0, val_x_t
                    del val_pred_pet_np, val_norm_pet_np, val_low_pet_np
                    del reversed_pred_pet, reversed_norm_pet, reversed_low_pet

                del max_v, low_dose_sino, norm_dose_sino, x_T, x_0, x_t, pred_x_0
                del residual_loss, pred_loss, total_loss

            diffusion.training_step = global_step

            if global_step % 10000 == 0:
                diffusion.save(save_path)
                        
            global_step = global_step + 1
            

if __name__ == '__main__':

    config = load_config("./config_ren.json")
    
    train_ren(params = config)

    print("done!")