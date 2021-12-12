import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

    
def compute_kde(xy_batch, crop_dim, downscale=4):
    """Computes a KDE with gaussian filters on a 2d array of cell locations, given in a batch of coordinates."""
    bs = len(xy_batch)
    kde = np.zeros((bs, 1, crop_dim//downscale, crop_dim//downscale))
    for i, xy in enumerate(xy_batch):
        for x, y in xy:
            x, y = int((x/downscale).round()), int((y/downscale).round())
            kde[i, :, y, x] = 1
        kde[i] = gaussian_filter(kde[i], sigma=30/downscale)
    return kde


def mask_fourier_tensor(fourier_in, xy_list, mask, cuda=False, device=None):
    """Mask the right entries of a fourier tensor. Expects the input to be a padded tensor."""
    bs = fourier_in.shape[0]
    fourier = fourier_in.clone()
    loc_mask = torch.zeros_like(fourier_in, dtype=bool)
    if cuda:
        loc_mask = loc_mask.cuda(device)
    for i in range(bs):
        for j, xy in enumerate(xy_list[i]):
            if mask[i, :, torch.round(xy[0]).long(), torch.round(xy[1]).long()] == 1:  # Might need to switch x and y
                fourier[i][j] = 0
                loc_mask[i,j] = 1
    return fourier, loc_mask


def mask_cpn(cpn, mask):  # REALLY SLOW ATM! (>3s for bs 128)
    """Remove the right entries in 'final_fourier' and 'xy' of a cpn dictionary object."""
    fourier = cpn["final_fourier"].copy()
    xy_list = cpn["xy"].copy()
    bs = len(xy_list)
    fourier_removed = []
    locs_removed = []
    for i in range(bs):
        pop_idx = []
        for j, xy in reversed(list(enumerate(xy_list[i]))):  # To pop the right indices reverse the list
            if mask[i, :, torch.round(xy[1]).long(), torch.round(xy[0]).long()] == 1:
                pop_idx.append(j)
                
        fourier_removed.append(fourier[i][pop_idx]) 
        fourier[i] = fourier[i][[k for k in range(len(xy_list[i])) if k not in pop_idx]]  # All indices but the ones in pop_idx
        locs_removed.append(xy_list[i][pop_idx])
        xy_list[i] = xy_list[i][[k for k in range(len(xy_list[i])) if k not in pop_idx]]
        
    cpn_masked = cpn.copy()
    cpn_masked["final_fourier"] = fourier
    cpn_masked["xy"] = xy_list
    return cpn_masked, fourier_removed, locs_removed


def pad_batch(batch_list, max_len=None, factor=1, f_dim=24, cuda=None, device=None, **kwargs):
    """Creates a Tensor of a batch of sequences from a list of sequences.
    All sequences are padded to the length of the longest one.
    Additionally returns a pad_mask with 1 for the padded entries.
    batch_list: list of tensors of shape [T, f_dim]."""
    bs = len(batch_list)
    lengths = [batch_item.shape[0] for batch_item in batch_list]
    if max_len is None:
        max_len = int(np.max(lengths)/factor+0.99)*factor  # Round to the next higher number divisible by factor
    
    batch_tensor = torch.empty(size=(bs, max_len, f_dim), **kwargs)
    pad_mask = torch.zeros_like(batch_tensor, dtype=bool)
    if cuda:
        batch_tensor = batch_tensor.cuda(device)
        pad_mask = pad_mask.cuda(device)
    
    for i, batch_item in enumerate(batch_list):
        batch_tensor[i] = F.pad(batch_item.reshape(1, -1, f_dim), (0,0,0,max_len-lengths[i]))
        pad_mask[i, lengths[i]:] = 1
    return batch_tensor, pad_mask
    
    
class RandomRotateFlip(torch.nn.Module):
    """Data augmentation class for the datasets. Performs random 90 degree rotations and h-/v-flips."""
    def __init__(self):
        super().__init__()
        self.angles = [-180, -90, 0, 90, 180]

    def __call__(self, imgs):
        hflip = np.random.rand() < 0.5
        vflip = np.random.rand() < 0.5
        rotate = np.random.rand() < 0.5
        if rotate:
            angle = random.choice(self.angles)
            imgs = [TF.rotate(img, angle) for img in imgs]
        if hflip:
            imgs = [TF.hflip(img) for img in imgs]
        if vflip:
            imgs = [TF.vflip(img) for img in imgs]   
        return imgs


def smoothen(array, ema_factor=0.1):
    """Exponential moving average smoothing for plots."""
    ema = np.array(array)
    for i in range(1, len(array)):
        ema[i] = ema[i-1] * (1-ema_factor) + ema[i] * ema_factor
    return ema


def hparam_plot_iterations(train_hist=None, test_hist=None, epochs=None, color="C0", bs=None,
                           label=None, xlabel=None, ylabel=None, ylim=None, ema=0.1):
    """Plotting function for hyperparameter tuning"""
    iterations = int(np.ceil(1000/bs))*epochs
    if len(test_hist) != epochs:
        train_hist = train_hist[:iterations]
        test_hist = test_hist[:epochs]
    
    if test_hist is not None:
        plt.scatter(np.arange(1, epochs+1)*iterations, test_hist, color=color, alpha=0.2)
        plt.plot(np.arange(1, epochs+1)*iterations, smoothen(test_hist, ema), color=color, label=label, linestyle="--")
    if train_hist is not None:
        plt.plot(smoothen(train_hist, ema), color=color, alpha=0.4)
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel, size=12)
    if ylabel is not None:
        plt.ylabel(ylabel, size=12)
    if ylim is not None:
        plt.ylim(ylim)


def hparam_plot_epochs(train_hist, test_hist, epochs, color="C0", bs=None,
                       label=None, xlabel=None, ylabel=None, ylim=None, ema=0.1):
    """Plotting function for hyperparameter tuning"""
    iterations = int(np.ceil(1000/bs))*epochs
    if len(test_hist) != epochs:
        train_hist = train_hist[:iterations]
        test_hist = test_hist[:epochs]
    examples = 1000
    plt.scatter(np.arange(1, epochs+1), test_hist, color=color, alpha=0.2)
    plt.plot(np.arange(1, epochs+1), smoothen(test_hist, ema), color=color, label=label, linestyle="--")
    plt.plot(np.arange(1, np.ceil(examples/bs)*epochs+1)/1000*bs, smoothen(train_hist, ema), color=color, alpha=0.4)
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel, size=12)
    if ylabel is not None:
        plt.ylabel(ylabel, size=12)
    if ylim is not None:
        plt.ylim(ylim)


def convert_to_uint8(tensor, inp_range=None):
    """Converts torch tensors or numpy arrays to uint8."""
    inp_type = type(tensor)
    if inp_range is None:
        inp_range = (tensor.min(), tensor.max())
    assert inp_type == torch.Tensor or inp_type == np.ndarray, "Input needs to be a torch.Tensor or numpy.ndarray."
    
    if inp_type == torch.Tensor:
        assert torch.min(tensor) >= inp_range[0] and torch.max(tensor) <= inp_range[1], f"Input tensor values need to be in [{inp_range[0]},{inp_range[1]}]."
    else:
        assert np.min(tensor) >= inp_range[0] and np.max(tensor) <= inp_range[1], f"Input tensor values need to be in [{inp_range[0]},{inp_range[1]}]."
 
    tensor = (tensor-inp_range[0])/(inp_range[1]-inp_range[0])*255+0.5  # +0.5 to round correctly when casting to uint8
    if inp_type == torch.Tensor:
        return tensor.type(torch.uint8)
    else:
        return tensor.astype(np.uint8)
    
    
def get_noise(loc, var, crop_dim, bs, min=None, max=None):
    """Helper function that returns (clamped) gaussian noise for a batch."""
    if min is not None or max is not None:
        return torch.normal(loc, var, size=(bs,1,crop_dim,crop_dim)).clamp(min=min, max=max)
    else:
        return torch.normal(loc, var, size=(bs,1,crop_dim,crop_dim))
