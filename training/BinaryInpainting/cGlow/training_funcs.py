import numpy as np
import time
import celldetection as cd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.ops import pad_batch, mask_fourier_tensor


def train(train_loader, test_loader, epochs, SG, SG_opt, SG_scheduler, DI, cpn, hparams, cuda, device, sample=True, test=False):
    """Training function for the cGlow model on Fourier data.
    Arguments:
        train_loader/test_loader: PyTorch DataLoader objects with training/test data
        epochs: Number of training epochs
        SG: PyTorch Neural Network class which is trained
        SG_opt: PyTorch optim object
        SG_scheduler: PyTorch optim.lr_scheduler object
        DI: DensityInpainting object
        cpn: CPN object
        hparams: List of required hyperparameters
        cuda: Whether to use cuda or not
        device: PyTorch device
        sample: Whether to plot samples every 10 epochs
        test: Whether to compute test loss after each epoch
    """
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    SG_loss_hist = []
    test_loss_hist = []
    crop_dim, order, f_dim, max_len, bs = hparams

    for epoch in range(epochs):
        start_time = time.time()
        for it, (crop, mask) in enumerate(train_loader):
            if cuda:
                crop = crop.cuda(device)
                mask = mask.cuda(device)

            # Get the outputs
#             locs, locs_in_mask = DI.fill_locations(cpn.inference(crop*(~mask)), mask, cuda=cuda, device=device)

            out = cpn.inference(crop)  # ~0.3s
            fourier_tgt = out["final_fourier"]
            xy = out["xy"]

            # Check for sequences longer than max_len 
            for i in range(len(fourier_tgt)-1,-1,-1):  # i in range(len(fourier_tgt)) but backwards
                if fourier_tgt[i].shape[0] >= max_len:
                    fourier_tgt.pop(i)
                    xy.pop(i)
                else:
                    fourier_tgt[i] = fourier_tgt[i].reshape(-1,f_dim)
#                     fourier_tgt[i] -= normalization  # Fourier normalization to 0 mean for each parameter respectively
            fourier_tgt, pad_mask = pad_batch(fourier_tgt, max_len=max_len, f_dim=f_dim, cuda=cuda, device=device)
            fourier_src, loc_mask = mask_fourier_tensor(fourier_tgt, xy, mask, cuda=cuda, device=device)  # ~0.7s
            fourier_tgt = fourier_tgt.unsqueeze(1)  # [bs, 1, max_len, f_dim]
            fourier_src = fourier_src.unsqueeze(1)  # [bs, 1, max_len, f_dim]
            
            z, nll = SG(fourier_src, fourier_tgt)
                    
            # Update the parameters
            SG_opt.zero_grad()
            loss = torch.mean(nll)
            scaler.scale(loss).backward()
#             if max_grad_clip > 0:
#                 torch.nn.utils.clip_grad_value_(SG.parameters(), max_grad_clip)
#             if max_grad_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(SG.parameters(), max_grad_norm)
            scaler.step(SG_opt)
            scaler.update()
            
            # Collect losses and accuracy
            SG_loss_hist.append(loss.item())
            
        if SG_scheduler is not None:
            SG_scheduler.step()
        
        # Test
        if test:
            SG.eval()
            test_loss = []
            for it, (crop, mask) in enumerate(test_loader):
                start = time.time()
                if cuda:
                    crop = crop.cuda(device)
                    mask = mask.cuda(device)

                # Get the outputs
                with torch.no_grad():
                    out = cpn.inference(crop)  # ~0.3s
                    fourier_tgt = out["final_fourier"]
                    xy = out["xy"]               
                # Check for sequences longer than max_len 
                for i in range(len(fourier_tgt)-1,-1,-1):  # i in range(len(fourier_tgt)) but backwards
                    if fourier_tgt[i].shape[0] >= max_len:
                        fourier_tgt.pop(i)
                        xy.pop(i)
                    else:
                        fourier_tgt[i] = fourier_tgt[i].reshape(-1,f_dim)
                with torch.no_grad():
                    fourier_tgt, pad_mask = pad_batch(fourier_tgt, max_len=max_len, f_dim=f_dim, cuda=cuda, device=device)
                    fourier_src, loc_mask = mask_fourier_tensor(fourier_tgt, xy, mask, cuda=cuda, device=device)  # ~0.7s
                    fourier_tgt = fourier_tgt.unsqueeze(1)  # [bs, 1, max_len, f_dim]
                    fourier_src = fourier_src.unsqueeze(1)  # [bs, 1, max_len, f_dim]
                    z, nll = SG(fourier_src, fourier_tgt)
                test_loss.append(torch.mean(nll).item())
            test_loss_hist.append(np.mean(test_loss))
            SG.train()
        
        if epoch%5 == 0:
            test_loss_mean = np.mean(test_loss_hist[-len(test_loader):]) if test else 0
            print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). G-Loss {np.mean(SG_loss_hist[-len(train_loader):]):3.2f}, Test-Loss {test_loss_mean:3.2f} \n")
            
        if sample and epoch%5 == 0:
            sample_SI(test_loader, SG, DI, cpn, hparams, seed=8, cuda=cuda, device=device)
            SG.train()
    if test:
        return SG_loss_hist, test_loss_hist
    else:
        return SG_loss_hist
        
        
def sample_SI(loader, SG, DI, cpn, hparams, cuda=False, device=None, plot=True, save_as=None, seed=None):
    """Sampling function for cGlow model on Fourier data.
    Arguments:
        loader: PyTorch DataLoader object
        SG: PyTorch Neural Network class which is trained
        DI: DensityInpainting object
        cpn: CPN object
        hparams: hparams: List of required hyperparameters
        cuda: Whether to use cuda or not
        device: PyTorch device
        plot: Whether to plot the results, else return the results
        save_as: Whether to save the plot
        seed: Set a random seed for reproducible results
    """
    start = time.time()
    SG.eval()
   
    crop_dim, order, f_dim, max_len, bs = hparams

    crop, mask = loader.dataset.__getitem__(0, seed=seed)
    mask = mask.unsqueeze(0)
    crop = crop.unsqueeze(0)

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
    
    if n_cells >= max_len:
        return sample_SI(loader, SG, plot=plot, cuda=cuda, device=device)
    
    locs_new = locs.clone()
    
    # Get Fourier Tensor 
    fourier_src = src["final_fourier"]
    fourier_src = fourier_src[0].reshape(1, -1, f_dim)
    fourier_src = fourier_src  # [1, S, f_dim]
    
    # Prepare Fourier Tensor for inference of new cells
    fourier_prd = F.pad(fourier_src, (0,0,0,n_new_cells))  # [1, T, f_dim]
    fourier_prd, pad_mask = pad_batch(fourier_prd, max_len=max_len, f_dim=f_dim, cuda=cuda, device=device)  # [1, max_len, f_dim]
    fourier_prd = fourier_prd.unsqueeze(1)  # Channels

    loc_mask = torch.zeros_like(fourier_prd, dtype=bool)  # [1, 1, max_len, f_dim]
    if cuda:
        loc_mask = loc_mask.cuda(device)
    loc_mask[:, :, n_cells-n_new_cells:n_cells] = 1

    with torch.no_grad():
        sample = SG(fourier_prd, y=None, reverse=True)[0]  # + normalization
    
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
        
        fourier_tgt = fourier_tgt.cpu().detach().numpy()
        fourier_prd = fourier_prd.squeeze(0).cpu().detach().numpy()
        
        n_nans = np.isnan(fourier_prd).sum()
        if n_nans > 0:
            print(f"{n_nans} NaNs found in the prediction, continue training.")
            return
        
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
        if save_as is not None:
            plt.savefig(save_as, dpi=200, bbox_inches='tight')
        plt.show()
    else:
        return fourier_prd
