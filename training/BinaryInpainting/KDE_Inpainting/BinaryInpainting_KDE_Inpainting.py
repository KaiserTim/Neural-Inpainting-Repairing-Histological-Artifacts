# %%

# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from utils.ops import smoothen
from models.models_GatedUNet import GatedUNet
from models.cpn_custom_forward_2 import CPN

from training_funcs import train, sample_DI


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

    wu = 8  # Width unit
    bs = 64
    lr = 2e-3
    wd = 0
    epochs = 100

    DG = GatedUNet(1, 1, wu, crop_dim, dilation=False, condition_channels=1, kernel_size=3, padding=1)

    if cuda:
        DG = DG.cuda(device)

    DG_opt = optim.Adam(DG.parameters(), lr=lr, betas=(0.9, 0.999))
    DG_scheduler = optim.lr_scheduler.MultiStepLR(DG_opt, [5, 10, 15, 25], 0.5)

    train_data = TrainSetIntact(1000, crop_dim, n_test=1, test=False)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    test_data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    print(f"Generator Number of Parameters: {DG.count_parameters()}")

    cpn = CPN(cuda, device)

    DG_loss_hist, test_loss_hist = train(train_loader, test_loader, epochs, DG, DG_opt, DG_scheduler, cpn, cuda, device,
                                         crop_dim, test=True)

    # torch.save(DG.state_dict(), f"saves/model_saves/DensityFill_GatedUNet_bs{bs}_lr{lr:.1e}_wu{wu}_wd{wd:.1e}_2.pt")
    # torch.save((DG_loss_hist, test_loss_hist), f"saves/model_saves/loss_hists_bs{bs}_lr{lr:.1e}_wu{wu}_wd{wd:.1e}_2.pt")

    bs = 64
    lr = 2e-3
    wu = 8
    wd = 0
    train_hist, test_hist = torch.load(f"model_saves/loss_hists/loss_hists_bs{bs}_lr{lr:.1e}_wu{wu}_wd{wd:.1e}_2.pt")
    iterations = int(1000 / bs)
    plt.figure(figsize=(16, 5))
    plt.scatter(np.arange(3, epochs + 1) * iterations, test_hist[2:], color=f"C{0}", alpha=0.2)
    plt.plot(np.arange(3, epochs + 1) * iterations, smoothen(test_hist[2:]), color=f"C{0}", label=f"lr {lr}")
    plt.plot(smoothen(train_hist), color=f"C{0}", alpha=0.3)
    plt.legend(loc=1)
    plt.xlabel(f"Iterations (bs {bs})")
    plt.ylabel("BCE Loss")
    plt.ylim([0, 0.2])
    plt.show()
