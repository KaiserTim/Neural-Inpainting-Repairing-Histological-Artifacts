# For imports from V3 folder
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import time
import numpy as np
import torch
import celldetection as cd

from torch.utils.data import DataLoader
from utils.datasets import TrainSetIntact
from utils.ops import get_noise, mask_cpn
from utils.II_noBI_evaluation import Evaluate
from models.models_GatedUNet import GatedUNet


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")

    def forward(G, crop, mask, cpn, cuda, device):
        bs, _, crop_dim, _ = mask.shape
        noise = get_noise(0, 1, crop_dim, bs)
        if cuda:
            noise = noise.cuda(device)

        crop_masked = crop * (1 - mask.half())
        G_out = G(crop_masked, condition=torch.cat((mask, noise), dim=1), tanh=True)
        G_painting = G_out * mask + crop * (1 - mask.half())

        cpn_out = cpn.inference(G_painting)
        cpn_masked, fourier_removed, locs_removed = mask_cpn(cpn_out, mask)  # Only the cells within the mask

        labels_prd = []
        for fourier, locs in zip(fourier_removed, locs_removed):
            contour = cd.data.cpn.fourier2contour(cd.asnumpy(fourier), cd.asnumpy(locs))
            labels_prd.append(cd.data.contours2labels(contour, mask[0].shape[-2:]))
        return labels_prd

    crop_dim = 256  # Quadratic dimension of the image crops
    order = 2  # Fourier-order hyperparameter

    bs = 16

    G = GatedUNet(1, 1, 16, crop_dim, dilation=True, condition_channels=2,
                  kernel_size=3, padding=1, padding_mode="reflect")
    G.eval()
    if cuda:
        G = G.cuda(device)

    G.load_state_dict(torch.load("saves/model_saves/G_GatedUNet_UNet_noBI.pt", map_location=device))

    data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    loader = DataLoader(data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {sum(p.numel() for p in G.parameters() if p.requires_grad)}")

    II_eval = Evaluate(G, forward, loader, cuda, device, order=order)

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CCE_results = II_eval.cell_count_error_fourier()
    torch.save(CCE_results, "saves/CCE_IInoBI_results")
    time.time() - start_time

    CCE_results = torch.load("saves/CCE_IInoBI_results")
    II_eval.cell_count_error_fourier(results=CCE_results, save_as="CCE_IInoBI");

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CSE_results = II_eval.cell_size_error_fourier()
    torch.save(CSE_results, "saves/CSE_IInoBI_results")
    time.time() - start_time

    CSE_results = torch.load("saves/CSE_IInoBI_results")
    II_eval.cell_size_error_fourier(results=CSE_results, save_as="CSE_IInoBI");

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CEE_results = II_eval.cell_eccentricity_error_fourier()
    torch.save(CEE_results, "saves/CEE_IInoBI_results")
    time.time() - start_time

    CEE_results = torch.load("saves/CEE_IInoBI_results")
    II_eval.cell_eccentricity_error_fourier(results=CEE_results, save_as="CEE_IInoBI");
