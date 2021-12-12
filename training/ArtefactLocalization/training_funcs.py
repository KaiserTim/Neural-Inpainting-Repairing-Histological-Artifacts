import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.ops import convert_to_uint8  # Only works when main package is in the python path


def train(train_loader, test_loader, epochs, predictor, p_opt, cuda, device, p_scheduler=None, sample=True, test=False):
    """Training function for Artifact Localization model.
    Arguments:
        train_loader/test_loader: PyTorch DataLoader objects with training/test data
        epochs: Number of training epochs
        predictor: PyTorch Neural Network class which is trained
        p_opt: PyTorch optim object
        cuda: Whether to use cuda or not
        device: PyTorch device
        p_scheduler: PyTorch optim.lr_scheduler object
        sample: Whether to plot samples every 10 epochs
        test: Whether to compute test loss after each epoch
    """
    scaler = torch.cuda.amp.GradScaler()
    train_start = time.time()
    criterion = nn.BCEWithLogitsLoss()
    P_loss_hist = []
    test_loss_hist = []
    for epoch in range(epochs):            
        start_time = time.time()        
        for it, (crop, anno) in enumerate(train_loader):  # Shapes: [bs, 1, crop_dim, crop_dim]
            if cuda:
                crop = crop.cuda(device)
                anno = anno.cuda(device)
                
            with torch.cuda.amp.autocast():
                anno_pred = predictor(crop)
            
            # Train the predictor
            p_opt.zero_grad()
            with torch.cuda.amp.autocast():
                P_loss = criterion(anno_pred, anno.half())
            scaler.scale(P_loss).backward()
            scaler.step(p_opt)
             
            scaler.update()

            P_loss_hist.append(P_loss.item())
            
        if p_scheduler is not None and epoch < 250:  # Set max epoch for decay
            p_scheduler.step()
        
        if test:
            predictor.eval()
            test_loss = []
            for crop, anno in test_loader:
                if cuda:
                    crop = crop.cuda(device)
                    anno = anno.cuda(device)
                with torch.cuda.amp.autocast():
                    anno_pred = predictor(crop)
                    P_loss = criterion(anno_pred, anno.half())
                test_loss.append(P_loss.item())
            test_loss_hist.append(np.mean(test_loss))
            predictor.train()
            
        if epoch%10 == 0:
            print(f"Epoch {epoch+1:2} done after {time.time()-start_time:2.1f}s ({(time.time()-train_start)/60:2.0f}m). P-Loss {np.mean(P_loss_hist[-len(train_loader):]):3.2f}, Test-Loss: {np.mean(test_loss_hist[-len(test_loader):]):3.2f}. \n")
            
        if sample and epoch%10 == 0: 
            with torch.cuda.amp.autocast():
                sample_P(test_loader, predictor, cuda=cuda, device=device)
            predictor.train()
            
    if test:
        return P_loss_hist, test_loss_hist
    else:
        return P_loss_hist


def sample_P(loader, predictor, cuda=False, device=None, plot=True, save_as=None, seed=None):
    """Sampling function for Artifact Localization model.
    Arguments:
        loader: PyTorch DataLoader object
        predictor: PyTorch Neural Network class which is trained
        cuda: Whether to use cuda or not
        device: PyTorch device
        plot: Whether to plot the results, else return the results
        save_as: Whether to save the plot
        seed: Set a random seed for reproducible results
    """
    predictor.eval()
    
    crop, anno = loader.dataset.__getitem__(0, seed=seed)
    crop = crop.unsqueeze(0)
    anno = anno.unsqueeze(0)
        
    if cuda:
        crop = crop.cuda(device)
        anno = anno.cuda(device)
        
    if seed is not None:
        torch.manual_seed(seed)
        
    with torch.no_grad():
        anno_pred = predictor(crop).sigmoid().round()
        
    pred = torch.round(torch.sigmoid(anno_pred))
    acc = pred.eq_(anno.half()).mean(dim=(-1, -2))
    acc = acc.squeeze(1).cpu().detach().numpy()
    print(f"Segmentation Accuracy: {acc[0]*100:.2f}%")
    
    if plot:
        anno = anno[0,0].cpu().detach().numpy().astype(np.bool_)
        crop = crop[0,0].cpu().detach().numpy().astype(np.single)
        anno_pred = anno_pred[0,0].cpu().detach().numpy()
        
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.title("Crop", size=15)
        plt.imshow(convert_to_uint8(crop), cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.title("Annotation", size=15)
        plt.imshow(convert_to_uint8(anno, inp_range=(0,1)), cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.title("Artifact Prediction", size=15)
        plt.imshow(convert_to_uint8(anno_pred, inp_range=(0,1)), cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        if save_as is not None:
            plt.savefig(save_as, dpi=200, bbox_inches='tight')
        plt.show()
    else:
        return anno_pred, pred, acc
