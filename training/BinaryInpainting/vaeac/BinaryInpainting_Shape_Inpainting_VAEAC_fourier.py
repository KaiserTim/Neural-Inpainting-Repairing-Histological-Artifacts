# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from models.cpn_custom_forward_2 import CPN
from models.density_fill_forward import DensityInpainting
from training_funcs import train, sample_SI

import models.vaeac.models_vaeac as models_vaeac
from models.vaeac.VAEAC import VAEAC


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    parallel = cuda and n_gpus > 1
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")

    crop_dim = 256  # Quadratic dimension of the image crops
    order = 2
    f_dim = 4*order

    bs = 32
    lr = 4e-4
    epochs = 100

    SG = VAEAC(models_vaeac.reconstruction_log_prob,
               models_vaeac.proposal_network,
               models_vaeac.prior_network,
               models_vaeac.generative_network)

    if cuda:
        SG = SG.cuda(device)

    SG_opt = torch.optim.Adam(SG.parameters(), lr=lr)

    train_data = TrainSetIntact(1000, crop_dim, n_test=1, test=False)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {sum(p.numel() for p in SG.parameters() if p.requires_grad)}")

    # The inits are required for separate GPU training
    cpn = CPN(cuda, device, order=order)
    DI = DensityInpainting(cuda, device)
    SG_loss_hist = train(train_loader, test_loader, epochs, SG, SG_opt, DI, cpn, order, cuda, device)

    # torch.save(SG.state_dict(), "saves/model_saves/ShapeInpainting_VAEAC_fourier_cnn_o2.pt")
    # torch.save(SG_loss_hist, "saves/model_saves/ShapeInpainting_VAEAC_fourier_cnn_o2_losshist.pt")

    plt.figure(figsize=(16,5))
    plt.suptitle(f"Final Loss: {SG_loss_hist[-1]:.2e}")
    plt.ylim([-0.01,0.1])
    plt.plot(SG_loss_hist)
    plt.show()
