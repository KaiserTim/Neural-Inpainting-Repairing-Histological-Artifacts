import torch
import time
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils.ops import get_noise, convert_to_uint8


def train(train_loader, test_loader, epochs, G, G_opt, D, D_opt, BI, cpn, cuda, device, G_scheduler=None, D_scheduler=None):
    """Training function for the Image Inpainting model.
        Arguments:
            train_loader/test_loader: PyTorch DataLoader objects with training/test data
            epochs: Number of training epochs
            G/D: PyTorch Neural Network class which is trained
            G_opt/D_opt: PyTorch optim object
            G_scheduler/D_scheduler: PyTorch optim.lr_scheduler object
            BI: BinaryInpainting object
            cpn: CPN object
            cuda: Whether to use cuda or not
            device: PyTorch device
    """
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    L1_loss = nn.L1Loss()
    BCE_loss = nn.BCEWithLogitsLoss()
    D_loss_hist = []
    D_acc_true_hist = []
    D_acc_fake_hist = []
    adv_loss_hist = []
    con_loss_hist = []
    
    for epoch in range(epochs):
        start_time = time.time()
        D_acc_true = []
        D_acc_fake = []
        pooling = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        for it, (crop, mask) in enumerate(train_loader):
            bs, _, crop_dim, _ = crop.shape
            noise = get_noise(0, 1, crop_dim, bs)
            if cuda:
                crop = crop.cuda(device)
                mask = mask.cuda(device)
                noise = noise.cuda(device)
            
            # Get the outputs
            with torch.cuda.amp.autocast():
                if cuda:
                    crop = crop.cuda(device)
                    mask = mask.cuda(device)
                    
                # Get the outputs
                crop_masked = crop*(1-mask.half())
                cpn_masked = cpn.inference(crop_masked+mask.half())  # Make the mask "white" to prevent cell detection
                
                with torch.no_grad():
                    BI_labels, pop_idx = BI.forward(cpn_masked, mask)
                BI_labels = torch.as_tensor(np.stack(BI_labels, axis=0)).unsqueeze(1)  # [bs, 1, crop_dim, crop_dim]
                if cuda:
                    BI_labels = BI_labels.cuda(device)
                BI_labels = 1 - 2*BI_labels  # Normalization to [-1,1]
                
                # Correct for removed examples
                keep_idx = [i for i in range(bs) if i not in pop_idx]  # test speed of this
                crop = crop[keep_idx]
                crop_masked = crop_masked[keep_idx]
                mask = mask[keep_idx]
                noise = noise[keep_idx]  # Initilaize noise here for optimization
                n_masked = torch.sum(mask)
                bs -= len(pop_idx)

                G_out = G(crop_masked, condition=torch.cat((mask, BI_labels, noise), dim=1))
                G_painting = G_out*mask + crop*(1-mask.half())
                D_out = D(G_painting, condition=BI_labels)
                
            # Train the generator
            G_opt.zero_grad()
            with torch.cuda.amp.autocast():
                edge = (mask.half() - (1-pooling(1-mask.half()))).bool()
                con_loss = L1_loss(G_out[edge], crop[edge])
                adv_loss = BCE_loss(D_out[mask], torch.zeros(n_masked).cuda(device))
                G_loss = adv_loss + con_loss
            scaler.scale(G_loss).backward()
            scaler.step(G_opt)
            
            # Train the discriminator
            D_opt.zero_grad()
            with torch.cuda.amp.autocast():
                D_out = D(G_painting.detach(), condition=BI_labels)
                D_loss = (BCE_loss(D_out[mask], torch.ones(n_masked).cuda(device))
                          +BCE_loss(D_out[~mask], torch.zeros(bs*crop_dim**2-n_masked).cuda(device)))/2
            scaler.scale(D_loss).backward()
            scaler.step(D_opt)
            scaler.update()
            
            # Collect losses and accuracy
            n_masked = torch.sum(mask)
            adv_loss_hist.append(adv_loss.item())
            con_loss_hist.append(con_loss.item())
            D_loss_hist.append(D_loss.item())
            D_acc_true.append((D_out[~mask].sigmoid().round() == torch.zeros(bs*crop_dim**2-n_masked).cuda(device)).float().mean().item())
            D_acc_fake.append((D_out[mask].sigmoid().round() == torch.ones(n_masked).cuda(device)).float().mean().item())
            
        D_acc_true_hist.append(np.mean(D_acc_true))
        D_acc_fake_hist.append(np.mean(D_acc_fake))
        
        if G_scheduler is not None:
            G_scheduler.step()
        if D_scheduler is not None:
            D_scheduler.step()
        
        print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). D-Loss {np.mean(D_loss_hist[-len(train_loader):]):3.2f}, True Acc: {np.mean(D_acc_true)*100:2.0f}%, Fake Acc: {np.mean(D_acc_fake)*100:2.0f}%, Adv-Loss {np.mean(adv_loss_hist[-len(train_loader):]):3.2f}, Con-Loss {np.mean(con_loss_hist[-len(train_loader):]):3.2f} \n")
        if epoch%2 == 0:
            with torch.cuda.amp.autocast():
                sample_II(test_loader, G, BI, cpn, cuda, device, seed=None)
    
    hists = {"D_loss_hist": D_loss_hist,
             "D_acc_true_hist": D_acc_true_hist, 
             "D_acc_fake_hist": D_acc_fake_hist, 
             "adv_loss_hist": adv_loss_hist, 
             "con_loss_hist": con_loss_hist} 
    return hists


def sample_II(loader, G, BI, cpn, cuda, device, plot=True, save_as=None, seed=None, noise=None):
    """Sampling function for Image Inpainting model.
        Arguments:
            loader: PyTorch DataLoader object
            G: PyTorch Neural Network class which is trained
            BI: BinaryInpainting object
            cpn: CPN object
            cuda: Whether to use cuda or not
            device: PyTorch device
            plot: Whether to plot the results, else return the results
            save_as: Whether to save the plot
            seed: Set a random seed for reproducible results, only applied to the crop and mask selection
            noise: Pass a noise tensor, instead of sampling one
    """
    G.eval()
    
    crop, mask = loader.dataset.__getitem__(0, seed=seed)
    bs, crop_dim, _ = crop.shape
    crop = crop.unsqueeze(1)
    mask = mask.unsqueeze(1)

    if noise is None:
        noise = get_noise(0, 1, crop_dim, 1)
        
    if cuda:
        crop = crop.cuda(device)
        mask = mask.cuda(device)
        noise = noise.cuda(device)
                                   
    crop_masked = crop*(1-mask.half())
    cpn_masked = cpn.inference(crop_masked+mask.half())  # Make the mask "white" to prevent cell detection

    with torch.no_grad():
        BI_labels, pop_idx = BI.forward(cpn_masked, mask)
    BI_labels = torch.as_tensor(np.stack(BI_labels, axis=0)).unsqueeze(1)  # [bs, 1, crop_dim, crop_dim]
    if cuda:
        BI_labels = BI_labels.cuda(device)
    BI_labels = 1 - 2*BI_labels  # Normalization to [-1,1]

    # Check for removed examples
    if len(pop_idx) > 0:
        print("Number of cells in the crop exceeded max_len")
        return

    G_out = G(crop_masked, condition=torch.cat((mask, BI_labels, noise), dim=1))
    G_painting = G_out*mask + crop*(1-mask.half())

    G.train()
    
    if plot:
        crop_masked = crop_masked[0,0].cpu().detach().numpy()
        G_painting = G_painting[0,0].cpu().detach().numpy().clip(-1,1)
        BI_labels = BI_labels[0,0].cpu().detach().numpy()
                
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.title("Crop Masked", size=12)        
        plt.imshow(convert_to_uint8(crop_masked, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)        
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.title("Painting", size=12)
        plt.imshow(convert_to_uint8(G_painting, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.title("Binary Painting", size=12)
        plt.imshow(convert_to_uint8(-BI_labels, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        if save_as is not None:
            plt.savefig(save_as, dpi=200, bbox_inches='tight')
        plt.show()
    else:
        return G_out, BI_labels
