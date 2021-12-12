import torch
import time
import torch.nn
import models.vaeac.models_vaeac as models_vaeac
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import celldetection as cd
import torch.nn.functional as F

from utils.ops import pad_batch, mask_fourier_tensor


def train(train_loader, test_loader, epochs, SG,  SG_opt, DI, cpn, order, cuda, device):
    """Training function for the VAEAC model on Fourier data.
        Arguments:
            train_loader/test_loader: PyTorch DataLoader objects with training/test data
            epochs: Number of training epochs
            SG: PyTorch Neural Network class which is trained
            SG_opt: PyTorch optim object
            DI: DensityInpainting object
            cpn: CPN object
            order: Order parameter for the Fourier Format
            cuda: Whether to use cuda or not
            device: PyTorch device
        """
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    f_dim = order*4
    SG_loss_hist = []
    
    for epoch in range(epochs):
        start_time = time.time()
        vlb_scale_factor = 128 ** 2
        for it, (crop, mask) in enumerate(train_loader):
            bs, _, crop_dim, _ = crop.shape
            if cuda:
                crop = crop.cuda(device)
                mask = mask.cuda(device)
                
            # Get the outputs
            with torch.cuda.amp.autocast():
                cpn_out = cpn.inference(crop)
                locs, locs_in_mask = DI.fill_locations(cpn_out, mask)  # ~0.3s
                
                fourier_tgt = cpn_out["final_fourier"]
                xy = cpn_out["xy"]
                # Check for sequences longer than 192
                flag = False
                for i in range(bs):
                    if fourier_tgt[i].shape[0] >= 192:
                         flag = True
                if flag:
                    continue
                    
                fourier_tgt, pad_mask = pad_batch(fourier_tgt, max_len=192, f_dim=f_dim, cuda=cuda, device=device)
                fourier_src, loc_mask = mask_fourier_tensor(fourier_tgt, xy, mask, cuda=cuda, device=device)  # ~0.7s
                fourier_tgt = fourier_tgt.unsqueeze(1)  # Channels
                fourier_src = fourier_src.unsqueeze(1)
                loc_mask = loc_mask.unsqueeze(1)
                pad_mask = pad_mask.unsqueeze(1)
                vlb = SG.batch_vlb(fourier_src, fourier_tgt, loc_mask, pad_mask).mean()   
                    
            # Upadate the parameters
            SG_opt.zero_grad()
            with torch.cuda.amp.autocast():
                SG_loss = -vlb / vlb_scale_factor
            scaler.scale(SG_loss).backward()
            scaler.step(SG_opt)
            scaler.update()
            
            # Collect losses and accuracy
            SG_loss_hist.append(SG_loss.item())
            
        # SG_scheduler.step()
        
        print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). G-Loss {np.mean(SG_loss_hist[-len(train_loader):]):3.2e} \n")
        if epoch%1 == 0:
            with torch.cuda.amp.autocast():
                sample_SI(test_loader, SG, DI, cpn, order, cuda, device, True)
            SG.train()
    return SG_loss_hist       


def sample_SI(loader, SG, DI, cpn, order, cuda=False, device=None, plot=True, seed=None):
    """Training function for the VAEAC model on Fourier data.
    Arguments:
        loader: PyTorch DataLoader object
        SG: PyTorch Neural Network class which is trained
        DI: DensityInpainting object
        cpn: CPN object
        order: Order parameter for the Fourier Format
        cuda: Whether to use cuda or not
        device: PyTorch device
        plot: Whether to plot the results, else return the results
        seed: Set a random seed for reproducible results
    """
    start = time.time()
    SG.eval()
    num_samples = 1
    f_dim = order*4
    
    crop, mask = loader.dataset.__getitem__(0, seed=seed)
    mask = mask.unsqueeze(0)
    crop = crop.unsqueeze(0)
    bs, _, crop_dim, _ = crop.shape

    if cuda:
        crop = crop.cuda(device)
        mask = mask.cuda(device)

    if seed is not None:
        torch.manual_seed(seed)
    
    with torch.no_grad():   
        src = cpn.inference(crop*(~mask))
        locs, locs_in_mask = DI.fill_locations(src, mask)
    locs = locs[0]  # [..., 2]
    locs_in_mask = locs_in_mask[0]  # [..., 2]
    n_new_cells = locs_in_mask.shape[0]
    n_cells = locs.shape[0]
    
    if n_cells >= 192:
        return sample_SI(loader, SG, DI, cpn, order, cuda, device, plot)
    
    locs_new = locs.clone()
    
    # Get Fourier Tensor 
    fourier_src = src["final_fourier"]
    fourier_src = fourier_src[0].reshape(1, -1, f_dim)
    fourier_src = fourier_src  # [1, S, f_dim]
    
    # Prepare Fourier Tensor for inference of new cells
    fourier_prd = F.pad(fourier_src, (0,0,0,n_new_cells))  # [1, T, f_dim]
    fourier_prd, pad_mask = pad_batch(fourier_prd, max_len=192, f_dim=f_dim, cuda=cuda, device=device)  # [1, 192, f_dim]
    fourier_prd = fourier_prd.unsqueeze(1)  # Channels
    pad_mask = pad_mask.unsqueeze(1)
    
    loc_mask = torch.zeros_like(fourier_prd, dtype=bool)  # [1, 1, 192, f_dim]
    if cuda:
        loc_mask = loc_mask.cuda(device)
    loc_mask[:, :, n_cells-n_new_cells:n_cells] = 1
    
    with torch.no_grad():
        sample_params = SG.generate_samples_params(fourier_prd, loc_mask, pad_mask, num_samples)[0].detach()
        
    sample = models_vaeac.sampler(sample_params)
    fourier_prd[loc_mask] = sample[loc_mask].float()
    fourier_prd = fourier_prd[:, :, :n_cells].reshape(1, -1, f_dim)
    print(f"Sampling done after {time.time()-start:2.1f}s")
    
    if plot:
        with torch.no_grad():
            tgt = cpn.inference(crop)  
        fourier_tgt = tgt["final_fourier"][0].reshape(1, -1, f_dim)
        _, loc_mask_tgt = mask_fourier_tensor(fourier_tgt, tgt["xy"], mask, cuda=cuda, device=device) 
        loc_mask_tgt = loc_mask_tgt.squeeze(1).cpu().detach().numpy()
        
        mask = mask[0,0].cpu().detach().numpy()
        loc_mask = loc_mask[0,0].cpu().detach().numpy()
        
        fourier_src = fourier_src.cpu().detach().numpy()
        fourier_tgt = fourier_tgt.cpu().detach().numpy()
        fourier_prd = fourier_prd.squeeze(0).cpu().detach().numpy()
        
        labels_src = cd.data.cpn.contours2labels(cd.asnumpy(src["contours"][0]), crop.shape[-2:])
        labels_tgt = cd.data.cpn.contours2labels(cd.asnumpy(tgt["contours"][0]), crop.shape[-2:])

        contours_prd = cd.data.cpn.fourier2contour(cd.asnumpy(fourier_prd.reshape(-1,order,4)), cd.asnumpy(locs_new))
        labels_prd = cd.data.cpn.contours2labels(contours_prd, crop.shape[-2:])

        fourier_prd_new = fourier_prd[loc_mask[:n_cells]].reshape(1,-1,f_dim)
        contours_prd_new = cd.data.cpn.fourier2contour(cd.asnumpy(fourier_prd_new.reshape(-1,order,4)), 
                                                       cd.asnumpy(locs_in_mask))
        labels_prd_new = cd.data.cpn.contours2labels(contours_prd_new, crop.shape[-2:])

        fourier_tgt_in_mask = fourier_tgt[loc_mask_tgt].reshape(1,-1,f_dim)
        contours_tgt_in_mask = cd.data.cpn.fourier2contour(cd.asnumpy(fourier_tgt_in_mask.reshape(-1,order,4)), 
                                                           cd.asnumpy(tgt["xy"][0][loc_mask_tgt[:,:,0]][:,[1,0]]))
        labels_tgt_in_mask = cd.data.cpn.contours2labels(contours_tgt_in_mask, crop.shape[-2:])

        plt.figure(figsize=(16,10))
        plt.subplot(2,3,1)
        plt.title("Ground Truth")
        plt.imshow(np.any(labels_tgt > 0, axis=-1), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.subplot(2,3,2)
        plt.title("Input")
        plt.imshow(np.any(labels_src > 0, axis=-1), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.subplot(2,3,3)
        plt.title("Prediction")
        plt.imshow(np.any(labels_prd > 0, axis=-1), vmin=0, vmax=1, cmap="gray")
        plt.imshow(mask, vmin=0, vmax=1, cmap="gray", alpha=0.15)
        plt.axis("off")
        plt.subplot(2,3,4)
        plt.title("Ground Truth New Cells")
        plt.imshow(np.any(labels_tgt_in_mask > 0, axis=-1), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.subplot(2,3,5)
        plt.title("Prediction Contour Outlines")
        cd.vis.show_detection(contours=contours_prd_new, cmap="gray", fill=0, contour_linestyle="-")
        down, up = plt.gca().get_ylim()
        left, right = plt.gca().get_xlim()
        plt.ylim([min(0, down), max(crop_dim, up)])
        plt.xlim([min(0, left), max(crop_dim, right)])
        plt.gca().invert_yaxis()  
        plt.subplot(2,3,6)
        plt.title("Prediction New Cells")
        plt.imshow(np.any(labels_prd_new > 0, axis=-1), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.show()
    else:
        return fourier_prd