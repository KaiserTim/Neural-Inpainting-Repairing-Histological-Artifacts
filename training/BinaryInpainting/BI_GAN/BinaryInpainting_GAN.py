# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from models.models_UNet import UNet
from models.models_GatedUNet import GatedUNet
from models.cpn_custom_forward_2 import CPN
from training_funcs import train, sample_BI


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

    bs = 16
    lr = 1e-4
    epochs = 100

    BG = GatedUNet(3, 1, 32, crop_dim, dilation=True, condition_channels=2, onehot=True,
              kernel_size=3, padding=1, padding_mode="reflect")
    BD = UNet(1, 1, 8, crop_dim, 1, leaky=True, batch_norm=False, sn=True,
              kernel_size=3, padding=1, padding_mode="reflect")

    if cuda:
        BG = BG.cuda(device)
        BD = BD.cuda(device)

    BG_opt = optim.Adam(BG.parameters(), lr=2*lr, betas=(0.5,0.999))
    BD_opt = optim.Adam(BD.parameters(), lr=lr, betas=(0.5,0.999))
    # BG_scheduler = optim.lr_scheduler.StepLR(BG_opt, 10, 0.9)
    # BD_scheduler = optim.lr_scheduler.StepLR(BD_opt, 10, 0.9)

    train_data = TrainSetIntact(100, crop_dim, n_test=True, test=False)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_data = TrainSetIntact(100, crop_dim, n_test=True, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {BG.count_parameters()}")
    print(f"Discriminator Number of Parameters: {BD.count_parameters()}")

    cpn = CPN(cuda, device, order=2)

    hists = train(train_loader, test_loader, epochs, BG, BD, BG_opt, BD_opt, cpn, cuda, device)

    # torch.save(BG.state_dict(), "model_saves/BG_GatedUNet_UNet_E72.pt")
    # torch.save(BD.state_dict(), "model_saves/BD_GatedUNet_UNet_E72.pt")

    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    plt.title("Losses")
    plt.plot(BD_loss_hist)
    plt.plot(adv_loss_hist)
    plt.plot(con_loss_hist)
    plt.legend(["D-Loss", "Adv-Loss", "Con-Loss", "G-Loss"])
    plt.ylim(bottom=0, top=4)
    plt.xlabel(f"Iterations (Batch-Size {bs})")
    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(BD_acc_true_hist)
    plt.plot(BD_acc_fake_hist)
    plt.legend(["True Acc", "Fake Acc"])
    plt.ylim([0,1])
    plt.xlabel(f"Epochs (1000 Examples per Epoch)")
    plt.show()
