# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from training_funcs import train, sample_SI
from models.cpn_custom_forward_2 import CPN
from models.density_fill_forward import DensityInpainting

from models.cGlow.CGlowModel import CondGlowModel


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")

    crop_dim = 256  # Quadratic dimension of the image crops
    order = 2  # Fourier-order hyperparameter
    f_dim = 4 * order
    max_len = 192  # Max sequence length for padding

    bs = 64
    lr = 5e-3
    epochs = 200

    SG_loss_hist = []

    x_size = (1, max_len, f_dim)
    y_size = (1, max_len, f_dim)
    x_hidden_channels = 128
    x_hidden_size = 64
    y_hidden_channels = 256
    K = 16
    L = 3
    learn_top = False
    y_bins = 2

    max_grad_clip = 5
    max_grad_norm = 0

    SG = CondGlowModel(x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, K, L, learn_top, y_bins)

    if cuda:
        SG = SG.cuda(device)

    SG_opt = torch.optim.Adam(SG.parameters(), lr=lr)
    SG_scheduler = optim.lr_scheduler.StepLR(SG_opt, 10, 0.9)

    train_data = TrainSetIntact(1000, crop_dim, n_test=1, test=False)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {sum(p.numel() for p in SG.parameters() if p.requires_grad)}")

    # The inits are required for separate GPU training
    cpn = CPN(cuda, device, order)
    DI = DensityInpainting(cuda, device)
    hparams = [crop_dim, order, f_dim, max_len, bs]

    SG_loss_hist = train(train_loader, test_loader, epochs, SG, SG_opt, SG_scheduler, DI, cpn, hparams,
                         cuda, device, sample=True, test=False)

    # torch.save(SG.state_dict(), f"saves/model_saves/ShapeInpainting_o{order}_ml{max_len}_bs{bs}_lr{lr:.1e}_lrdc.pt")
    # torch.save(SG_loss_hist, f"saves/model_saves/loss_hists_o{order}_ml{max_len}_bs{bs}_lr{lr:.1e}_lrdc.pt")

    plt.figure(figsize=(16, 5))
    plt.title(f"Final Loss: {SG_loss_hist[-1]:.2e}")
    plt.plot(SG_loss_hist)
    plt.ylim([0, 2])
    plt.show()

