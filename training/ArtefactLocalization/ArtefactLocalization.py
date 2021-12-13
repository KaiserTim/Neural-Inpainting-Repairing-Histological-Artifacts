# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainSetArtefacts
from utils.ops import hparam_plot_epochs
from models.models_UNet import UNet
from training_funcs import train, sample_P


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")

    crop_dim = 384  # Quadratic dimension of the image crops

    wu = 16  # Width unit
    bs = 16
    lr = 4e-4
    epochs = 300
    wd = 5e-3

    predictor = UNet(1, 1, wu, crop_dim, 1, dilation=False, kernel_size=3, padding=1, bias=False, padding_mode="reflect")
    print(f"Predictor Number of Paramters: {predictor.count_parameters():,}")

    if cuda:
        predictor = predictor.cuda(device)

    P_opt = optim.Adam(predictor.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    P_scheduler = optim.lr_scheduler.StepLR(P_opt, 10, 0.9)

    train_data = TrainSetArtefacts(1000, crop_dim, n_test=10)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_data = TrainSetArtefacts(100, crop_dim, n_test=10, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    P_loss_hist, test_loss_hist = train(train_loader, test_loader, epochs, predictor, P_opt, cuda, device, P_scheduler,
                                        sample=False, test=True)


    plt.figure(figsize=(16, 5))
    plt.title("Artifact Localization Loss", size=12)
    hparam_plot_epochs(P_loss_hist, test_loss_hist, epochs, f"C{0}", bs, lr, wd, wu, label=f"lr {lr:.0e}",
                       xlabel=f"Epochs", ylabel="BCE Loss", ylim=[0, 0.25])
    plt.show()

    # torch.save(predictor.state_dict(), "model_saves/predictor.pt")
    # torch.save((P_loss_hist, test_loss_hist), f"model_saves/loss_hists_predictor.pt")
