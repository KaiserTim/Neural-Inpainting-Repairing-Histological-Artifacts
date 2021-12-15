# Use if main package is not in python path
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from models.models_GatedUNet import GatedUNet
from models.BinaryInpainting_forward import BinaryInpainting
from models.cpn_custom_forward_2 import CPN
from training_funcs import train, sample_II


# Note on this model: The training results are inconsistent and very sensitive to hyperparameter changes.
# $\beta_1 \leq 0.8$ in the Adam optimizers seems to very important for training stability in runs longer than 20-30
# epochs. The results are less sensitive to small changes in width unit or learning rates. It is unclear what causes
# artifacting to appear in specific examples and how to prevent it.

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")

    crop_dim = 256  # Quadratic dimension of the image crops (should be divisible by 32 to avoid cuda error with demics/pad)

    bs = 32
    lr = 1e-4
    epochs = 200

    G = GatedUNet(1, 1, 32, crop_dim, dilation=True, condition_channels=3,
                  kernel_size=3, padding=1, padding_mode="reflect")
    D = GatedUNet(1, 1, 32, crop_dim, condition_channels=1, leaky=True, sn=True,
                  kernel_size=3, padding=1, padding_mode="reflect")

    if cuda:
        G = G.cuda(device)
        D = D.cuda(device)

    G_opt = optim.Adam(G.parameters(), lr=4 * lr, betas=(0.5, 0.999))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    train_data = TrainSetIntact(1000, crop_dim, n_test=1, test=False)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {G.count_parameters():,}")
    print(f"Discriminator Number of Parameters: {D.count_parameters():,}")

    cpn = CPN(cuda, device, order=2)
    BI = BinaryInpainting(cuda, device, max_len=192)

    hists = train(train_loader, test_loader, epochs, G, G_opt, D, D_opt, BI, cpn, cuda, device)

    # torch.save(G.state_dict(), "model_saves/G_GatedUNet_GatedUNet.pt")
    # torch.save(D.state_dict(), "model_saves/D_GatedUNet_GatedUNet.pt")
    # torch.save(hists, "model_saves/loss_hists/GatedUNet_GatedUNet_hists.pt")

    D_loss_hist = hists["D_loss_hist"]
    D_acc_true_hist = hists["D_acc_true_hist"]
    D_acc_fake_hist = hists["D_acc_fake_hist"]
    adv_loss_hist = hists["adv_loss_hist"]
    con_loss_hist = hists["con_loss_hist"]

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Losses")
    plt.plot(D_loss_hist)
    plt.plot(adv_loss_hist)
    plt.plot(con_loss_hist)
    plt.legend(["D-Loss", "Adv-Loss", "Con-Loss"])
    plt.ylim(bottom=0)
    plt.xlabel(f"Iterations (Batch-Size {bs})")
    plt.ylim([0, 5])
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(D_acc_true_hist)
    plt.plot(D_acc_fake_hist)
    plt.plot(np.ones(len(D_acc_true_hist)) * 0.5, linestyle=":", color="black")
    plt.legend(["True Acc", "Fake Acc"])
    plt.ylim([0, 1])
    plt.xlabel(f"Epochs ({1000} Examples per Epoch)")
    # plt.savefig("imgs/GatedUNet_GatedUnet")
    plt.show()
