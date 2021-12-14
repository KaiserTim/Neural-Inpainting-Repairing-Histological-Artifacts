import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import celldetection as cd

from utils.ops import convert_to_uint8, get_noise


def train(train_loader, test_loader, epochs, BG, BD, BG_opt, BD_opt, cpn, cuda, device):
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    BCE_loss = nn.BCEWithLogitsLoss()
    pooling = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
    
    BG_loss_hist = []
    BD_loss_hist = []
    BD_acc_true_hist = []
    BD_acc_fake_hist = []
    adv_loss_hist = []
    con_loss_hist = []
    hists = [BG_loss_hist, BD_loss_hist, BD_acc_true_hist, BD_acc_fake_hist, adv_loss_hist, con_loss_hist]
    
    for epoch in range(epochs):
        start_time = time.time()
        BD_acc_true = []
        BD_acc_fake = []
        for it, (crop, mask) in enumerate(train_loader):
            bs, _, crop_dim, _ = crop.shape
            noise = get_noise(0, 1, crop_dim, bs)
            if cuda:
                crop = crop.cuda(device)
                mask = mask.cuda(device)
                noise = noise.cuda(device)
            n_masked = torch.sum(mask)
            
            # Get the outputs
            with torch.cuda.amp.autocast():
                cpn_out = cpn.inference(crop)
                labels = []
                contours = cpn_out["contours"]  # List of contours
                for contour in contours:
                    labels.append(np.any(cd.data.contours2labels(contour.cpu().data.numpy(), crop[0].shape[-2:]) > 0, axis=-1))
                segment = torch.as_tensor(np.stack(labels)).unsqueeze(1).long()
                if cuda:
                    segment = segment.cuda(device)
                segment_masked = segment*(1-mask.long()) + 2*mask.long()  # Cut out the masked area
                
                BG_out = BG(segment_masked, condition=torch.cat((mask, noise), dim=1), sigmoid=False)
                BG_painting = torch.sigmoid(BG_out)*mask + segment*(1-mask.float())
                BD_out = BD(BG_painting)
                
            # Train the generator
            BG_opt.zero_grad(), BG.zero_grad()
            with torch.cuda.amp.autocast():
                edge = (mask.float() - (1-pooling(1-mask.float()))).bool()
                con_loss = BCE_loss(BG_out[edge], segment[edge].float())
                adv_loss = BCE_loss(BD_out[mask], torch.zeros(n_masked).cuda(device))
                BG_loss = adv_loss + con_loss/5
            scaler.scale(BG_loss).backward()
            scaler.step(BG_opt)
            
            # Train the discriminator
            BD_opt.zero_grad(), BD.zero_grad()
            with torch.cuda.amp.autocast():
                BD_out = BD(BG_painting.detach())
                BD_loss = (BCE_loss(BD_out[mask], torch.ones(n_masked).cuda(device))
                          +2*BCE_loss(BD_out[~mask], torch.zeros(bs*crop_dim**2-n_masked).cuda(device)))/3
            scaler.scale(BD_loss).backward()
            scaler.step(BD_opt)
            scaler.update()
            
            # Collect losses and accuracy
            n_masked = torch.sum(mask)
            BG_loss_hist.append(BG_loss.item())
            BD_loss_hist.append(BD_loss.item())
            adv_loss_hist.append(adv_loss.item())
            con_loss_hist.append(con_loss.item())
            BD_acc_true.append((BD_out[~mask].sigmoid().round() == torch.zeros(bs*crop_dim**2-n_masked).cuda(device)).float().mean().item())
            BD_acc_fake.append((BD_out[mask].sigmoid().round() == torch.ones(n_masked).cuda(device)).float().mean().item())
        BD_acc_true_hist.append(np.mean(BD_acc_true))
        BD_acc_fake_hist.append(np.mean(BD_acc_fake))
        
#         BG_scheduler.step()
#         BD_scheduler.step()        
        
        print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). D-Loss {np.mean(BD_loss_hist[-len(train_loader):]):3.2f}, True Acc: {np.mean(BD_acc_true)*100:2.0f}%, Fake Acc: {np.mean(BD_acc_fake)*100:2.0f}%, Adv-Loss {np.mean(adv_loss_hist[-len(train_loader):]):3.2f}, Con-Loss {np.mean(con_loss_hist[-len(train_loader):]):3.2f} \n")
        if epoch%1 == 0:
            with torch.cuda.amp.autocast() and torch.no_grad():
                sample_training(test_loader, BG, BD, cpn, cuda, device, epoch, crop_dim)
            BG.train()
            BD.train()
    return hists


def sample_training(loader, BG, BD, cpn, cuda, device, epoch, crop_dim, plot=True, seed=None):
    BG.eval()
    BD.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
        
    noise = get_noise(0, 1, crop_dim, 3)
        
    crop = torch.zeros(3, 1, crop_dim, crop_dim)
    mask = torch.zeros(3, 1, crop_dim, crop_dim, dtype=torch.bool)
    seeds = [2, 0, 1]
    for i in range(3):
        c, m = loader.dataset.__getitem__(0, seed=seeds[i])
        crop[i] = c
        mask[i] = m

    if cuda:
        crop = crop.cuda(device)
        mask = mask.cuda(device)
        noise = noise.cuda(device)
        
    cpn_out = cpn.inference(crop)
    labels = []
    contours = cpn_out["contours"]  # List of contours
    for contour in contours:
        labels.append(np.any(cd.data.contours2labels(contour.cpu().data.numpy(), crop[0].shape[-2:]) > 0, axis=-1))
    segment = torch.as_tensor(np.stack(labels)).unsqueeze(1).long()
    if cuda:
        segment = segment.cuda(device)
    segment_masked = segment*(1-mask.long()) + 2*mask.long()  # Cut out the masked area

    BG_out = BG(segment_masked, condition=torch.cat((mask, noise), dim=1), sigmoid=True)
    # BG_painting = BG_out*mask + segment*(1-mask.float())
    # BD_out = BD(BG_painting, sigmoid=True)
    
    if plot:
        mask = mask.squeeze(1).cpu().detach().numpy()
        segment = segment.squeeze(1).cpu().detach().numpy()
        BG_out = BG_out.squeeze(1).cpu().detach().numpy()
                            
        plt.figure(figsize=(16,10))
        plt.subplot(2,3,1)
        plt.title("Segment 1")
        plt.imshow(convert_to_uint8(segment[0]*mask[0]+(1-mask[0])*0.5, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,2)
        plt.title("Segment 2")
        plt.imshow(convert_to_uint8(segment[1]*mask[1]+(1-mask[1])*0.5, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,3)
        plt.title("Segment 3")
        plt.imshow(convert_to_uint8(segment[2]*mask[2]+(1-mask[2])*0.5, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,4)
        plt.title("Binary Painting", size=12)
        plt.imshow(convert_to_uint8(BG_out[0]*mask[0], inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,5)
        plt.title("Generator Out 2")
        plt.imshow(convert_to_uint8(BG_out[1]*mask[1], inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,6)
        plt.title("Generator Out 3")
        plt.imshow(convert_to_uint8(BG_out[2]*mask[2], inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.show()
    else:
        return BG_out
    
    
def sample_BI(data, BG, BD, cpn, plot=True, save_as=None, seed=None, cuda=False, device=None):
    BG.eval()
    BD.eval()
    
    crop, mask = data.__getitem__(0)
    mask = mask.unsqueeze(0)
    crop = crop.unsqueeze(0)
    crop_dim = crop.shape[-1]
    noise = get_noise(0, 1, crop_dim, 1)

    if cuda:
        crop = crop.cuda(device)
        mask = mask.cuda(device)
        noise = noise.cuda(device)

    if seed is not None:
        torch.manual_seed(seed)
        
    cpn_out = cpn.inference(crop)
    labels = []
    contours = cpn_out["contours"]  # List of contours
    for contour in contours:
        labels.append(np.any(cd.data.contours2labels(contour.cpu().data.numpy(), crop[0].shape[-2:]) > 0, axis=-1))
    segment = torch.as_tensor(np.stack(labels)).unsqueeze(1).long()
    if cuda:
        segment = segment.cuda(device)
    segment_masked = segment*(1-mask.long()) + 2*mask.long()  # Cut out the masked area
    crop_masked = crop*(1-mask.half()) + 2*mask.half()
    
    BG_out = BG(segment_masked, condition=torch.cat((mask, noise), dim=1), sigmoid=True)
    BG_painting = BG_out*mask + segment*(1-mask.float())
    BD_out = BD(BG_painting, sigmoid=True)
    
    if plot:
        crop = crop[0,0].cpu().detach().numpy()
        mask = mask[0,0].cpu().detach().numpy()
        segment = segment[0,0].cpu().detach().numpy()
        segment_masked = segment_masked[0,0].cpu().detach().numpy()
        BG_out = BG_out[0,0].cpu().detach().numpy()
        BG_painting = BG_painting[0,0].cpu().detach().numpy()
        BD_out = BD_out[0,0].cpu().detach().numpy()
                
        plt.figure(figsize=(16,10))
        plt.subplot(2,3,1)
        plt.title("Crop")
        plt.imshow(convert_to_uint8(crop, inp_range=(-1,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,2)
        plt.title("Segmentation")
        plt.imshow(convert_to_uint8(segment, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,3)
        plt.title("Mask")
        plt.imshow(convert_to_uint8(mask, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,4)
        plt.title("Generator Output")
        plt.imshow(convert_to_uint8(BG_out, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,5)
        plt.title("Generator Reconstruction")
        plt.imshow(convert_to_uint8(BG_painting, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(2,3,6)
        plt.title("Discriminator Output")
        plt.imshow(convert_to_uint8(BD_out, inp_range=(0,1)), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        if save_as is not None:
            plt.savefig(save_as, dpi=200)
        plt.show()