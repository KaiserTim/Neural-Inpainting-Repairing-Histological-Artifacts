# For imports from V3 folder
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import time
import numpy as np
import torch

from utils.DI_evaluation import Evaluate
from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact


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
    f_dim = 4*order
    max_len = 192  # Max sequence length for padding

    bs = 16

    data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    loader = DataLoader(data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    DI_eval = Evaluate(loader, cuda, device, order)

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CCE_results = DI_eval.cell_count_error_fourier()
    torch.save(CCE_results, "saves/CCE_results")
    time.time()-start_time

    CCE_results = torch.load("saves/CCE_results")
    DI_eval.cell_count_error_fourier(results=CCE_results, save_as="CCE_DI");

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CKE_results = DI_eval.cell_kde_error_fourier()
    torch.save(CKE_results, "saves/CKE_results")
    time.time()-start_time

    CKE_results = torch.load("saves/CKE_results")
    DI_eval.cell_kde_error_fourier(results=CKE_results, save_as="CKE_DI");
