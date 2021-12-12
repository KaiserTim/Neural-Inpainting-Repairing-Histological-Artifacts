import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import celldetection as cd

from torchvision.transforms.functional import resize
from utils.ops import convert_to_uint8, compute_kde, mask_cpn


def train_forward(DG, crop, mask, cpn, cuda, device, crop_dim, downscale):
    """Forward pass for the one iteration in the training loop."""
    cpn_true = cpn.inference(crop)
    xy_true = cd.asnumpy(cpn_true["xy"])  # list of ndarrays of shape [n_cells, 2]
    kde_true = torch.as_tensor(compute_kde(xy_true, crop_dim, downscale=downscale),
                               dtype=torch.half)  # [bs, 1, crop_dim/downscale, crop_dim/downscale]
    kde_true = (kde_true - 0.02635) * 18  # Normalize approximately to [-1,1] and 0 mean

    cpn_masked = mask_cpn(cpn_true, mask)[0]  # Very slow
#                 cpn_masked = cpn.inference(crop*(~mask))
    xy_masked = cd.asnumpy(cpn_masked["xy"])  # list of ndarrays of shape [n_cells, 2]
    kde_masked = torch.as_tensor(compute_kde(xy_masked, crop_dim, downscale=downscale),
                                 dtype=torch.half)  # [bs, 1, crop_dim/downscale, crop_dim/downscale]
    if cuda:
        kde_true = kde_true.cuda(device)
        kde_masked = kde_masked.cuda(device)

    mask_small = resize(mask, (crop_dim//downscale, crop_dim//downscale))
    kde_masked = (kde_masked - 0.02635) * 18  # Normalize approximately to [-1,1] and 0 mean
    kde_masked = kde_masked*(1-mask_small.half())  # Remove the masked area
    DG_out = DG(kde_masked, condition=mask_small.half(), tanh=True)
    return DG_out, kde_true, kde_masked, mask_small


def train(train_loader, test_loader, epochs, DG, DG_opt, DG_scheduler, cpn, cuda, device, crop_dim, downscale=4, sample=True, test=False):
    """Training function for the DensityInpainting model.
        Arguments:
            train_loader/test_loader: PyTorch DataLoader objects with training/test data
            epochs: Number of training epochs
            DG: PyTorch Neural Network class which is trained
            DG_opt: PyTorch optim object
            DG_scheduler: PyTorch optim.lr_scheduler object
            cpn: CPN object
            cuda: Whether to use cuda or not
            device: PyTorch device
            crop_dim: Dimension of the input crops
            downscale: Factor for downscaling the KDE image along each axis
            test: Whether to compute test loss after each epoch
            sample: Whether to plot samples every 10 epochs
        """
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    L1_loss = nn.L1Loss()
    DG_loss_hist = []
    test_loss_hist = []
    
    for epoch in range(epochs):
        start_time = time.time()
        for it, (crop, mask) in enumerate(train_loader):
            if cuda:
                crop = crop.cuda(device)
                mask = mask.cuda(device)

            # Get the outputs
            with torch.cuda.amp.autocast():
                DG_out, kde_true, kde_masked, mask_small = train_forward(DG, crop, mask, cpn, cuda, device, crop_dim, downscale)
                
            # Train the generator
            DG_opt.zero_grad()
            with torch.cuda.amp.autocast():
                DG_loss = L1_loss(DG_out[mask_small], kde_true[mask_small])
            scaler.scale(DG_loss).backward()
            scaler.step(DG_opt)
            scaler.update()
            
            # Collect losses and accuracy
            DG_loss_hist.append(DG_loss.item())
        if DG_scheduler is not None:
            DG_scheduler.step()
        
        if test:
            DG.eval()
            test_loss = []
            for crop, mask in test_loader:
                if cuda:
                    crop = crop.cuda(device)
                    mask = mask.cuda(device)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        DG_out, kde_true, kde_masked, mask_small = train_forward(DG, crop, mask, cpn, cuda, device, crop_dim, downscale)
                        DG_loss = L1_loss(DG_out[mask_small], kde_true[mask_small])
                test_loss.append(DG_loss.item())
            test_loss_hist.append(np.mean(test_loss))
            DG.train()
            
        if epoch%5 == 0:
            print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). G-Loss {np.mean(DG_loss_hist[-len(train_loader):]):4.3f}, Test-Loss: {np.mean(test_loss_hist[-len(test_loader):]):4.3f}. \n")
            
        if sample and epoch%5 == 0:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    sample_DI(test_loader, DG, cpn=cpn, crop_dim=crop_dim, downscale=downscale, plot=True, cuda=cuda, device=device)
            DG.train()
    
    if test:
        return DG_loss_hist, test_loss_hist
    else:
        return DG_loss_hist
            
            
def sample_DI(loader, DG, cpn, crop_dim, downscale=4, cuda=False, device=None, plot=True, save_as=None, seed=None):
    """Sampling function for DensityInpainting model.
        Arguments:
            loader: PyTorch DataLoader object
            DG: PyTorch Neural Network class which is trained
            cpn: CPN object
            crop_dim: Dimension of the input crops
            downscale: Factor for downscaling the KDE image along each axis
            cuda: Whether to use cuda or not
            device: PyTorch device
            plot: Whether to plot the results, else return the results
            save_as: Whether to save the plot
            seed: Set a random seed for reproducible results
        """
    DG.eval()
    
    crop, mask = loader.dataset.__getitem__(0)
    mask = mask.unsqueeze(0)
    crop = crop.unsqueeze(0)

    if cuda:
        crop = crop.cuda(device)
        mask = mask.cuda(device)

    if seed is not None:
        torch.manual_seed(seed)
        
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            DG_out, kde_true, kde_masked, mask_small = train_forward(DG, crop, mask, cpn, cuda, device, crop_dim, downscale)
    
    if plot:
        DG_out = DG_out.cpu().detach()
        mask_small = mask_small.cpu().detach()
        kde_true = kde_true.cpu().detach()
        DG_painting = DG_out*mask_small + kde_true*(~mask_small)
        
        crop = crop[0,0].cpu().detach().numpy()
        mask_small = mask_small[0,0].numpy()
        kde_true = torch.clamp(kde_true[0,0].float(), min=-1, max=1).numpy()
        kde_masked = torch.clamp(kde_masked[0,0].float(), min=-1, max=1).cpu().detach().numpy()
        DG_out = torch.clamp(DG_out[0,0].float(), min=-1, max=1).numpy()
        DG_painting = torch.clamp(DG_painting[0,0].float(), min=-1, max=1).cpu().detach().numpy()
#         xy = cpn_masked["xy"][0].cpu().detach().numpy()
    
        tmp1 = kde_true/18 + 0.02635
        tmp2 = DG_out/18 + 0.02635
        print(f"KDE mask volume: {tmp1[mask_small].sum():.2f}, Out mask volume: {tmp2[mask_small].sum():.2f}")
        
        plt.figure(figsize=(16,10))
        plt.subplot(2,3,1)
        plt.title("Crop")
        plt.imshow(convert_to_uint8(crop, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
#         plt.scatter(xy[:,0], xy[:,1], color="yellow")
        plt.axis("off")
        plt.subplot(2,3,2)
        plt.title("Ground Truth KDE")
        plt.imshow(convert_to_uint8(kde_true, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,3)
        plt.title("Ground Truth Mask")
        plt.imshow(convert_to_uint8(kde_true*mask_small + (1-mask_small), inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,4)
        plt.title("Generator Output")
        plt.imshow(convert_to_uint8(DG_out, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,5)
        plt.title("Generator Painting")
        plt.imshow(convert_to_uint8(DG_painting, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,6)
        plt.title("Generator Output Mask")
        plt.imshow(convert_to_uint8(DG_out*mask_small + (1-mask_small), inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        if save_as is not None:
            plt.savefig(save_as, dpi=200, bbox_inches='tight')
        plt.show()
    else:
        return DG_out
