# For imports from V3 folder
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import time
import numpy as np
import torch
import celldetection as cd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.density_fill_forward import DensityInpainting
from utils.datasets import TrainSetIntact
from utils.ops import pad_batch
from utils.BI_evaluation import Evaluate

import models.vaeac.models_vaeac as models_vaeac
from models.vaeac.VAEAC import VAEAC


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42);

    n_gpus = torch.cuda.device_count()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using GPU: {cuda}")
    print(f"Available GPUs: {n_gpus}")
    print("Only tested with CUDA enabled")


    def forward(SG, cpn_masked, mask, cuda, device, return_fourier=False):
        bs = mask.shape[0]
        with torch.no_grad():
            locs, locs_in_mask = DI.fill_locations(cpn_masked, mask)
        n_new_cells = [locs_in_mask[i].shape[0] for i in range(len(locs_in_mask))]
        n_cells = [locs[i].shape[0] for i in range(len(locs))]

        # Get Fourier Tensor
        fourier_prd = cpn_masked["final_fourier"]
        order = fourier_prd[0].shape[-2]
        f_dim = order * 4
        pop_idx = []  # list of indices with crops with n_cells > max_len
        for i in range(bs - 1, -1, -1):  # range(bs) but backwards
            if n_cells[i] >= max_len:
                pop_idx.append(i)
                fourier_prd.pop(i)
                n_cells.pop(i)
                n_new_cells.pop(i)
                locs.pop(i)
                locs_in_mask.pop(i)
            else:
                fourier_prd[i] = fourier_prd[i].reshape(-1, f_dim)
                # Prepare Fourier Tensor for inference of new cells
                fourier_prd[i] = F.pad(fourier_prd[i], (0, 0, 0, n_new_cells[i]))  # [T, f_dim]
        bs -= len(pop_idx)

        fourier_prd, pad_mask = pad_batch(fourier_prd, max_len=max_len, f_dim=f_dim, cuda=cuda,
                                          device=device)  # [bs, max_len, f_dim]
        fourier_prd = fourier_prd.unsqueeze(1)  # Channel dimension
        pad_mask = pad_mask.unsqueeze(1)

        loc_mask = torch.zeros_like(fourier_prd, dtype=bool)  # [bs, 1, max_len, f_dim]
        if cuda:
            loc_mask = loc_mask.cuda(device)
        for i in range(bs):
            loc_mask[i, :, n_cells[i] - n_new_cells[i]:n_cells[i]] = 1  # Make this parallelized?

        with torch.no_grad():
            sample_params = SG.generate_samples_params(fourier_prd, loc_mask, pad_mask)[:, 0].detach()
        sample = models_vaeac.sampler(sample_params)  # Mean sampler atm

        fourier_prd[loc_mask] = sample[loc_mask].float()
        if return_fourier:
            fourier_model = []
            for i in range(bs):
                fourier_model.append(fourier_prd[i][loc_mask[i]])
            return fourier_model, pop_idx  # Return only the new cells

        labels_prd = []
        for i in range(bs):
            # prd = fourier_prd[i, :, :n_cells[i]].reshape(-1,order,4)  # All cells
            prd = fourier_prd[i, :, n_cells[i] - n_new_cells[i]:n_cells[i]].reshape(-1, order, 4)  # Only the new cells
            contour = cd.data.cpn.fourier2contour(cd.asnumpy(prd), cd.asnumpy(locs_in_mask[i]))
            labels_prd.append(cd.data.contours2labels(contour, mask.shape[-2:]))
        return labels_prd, pop_idx


    crop_dim = 256  # Quadratic dimension of the image crops
    order = 2  # Fourier-order hyperparameter
    f_dim = 4 * order
    max_len = 192  # Max sequence length for padding

    bs = 16

    SG = VAEAC(models_vaeac.reconstruction_log_prob,
               models_vaeac.proposal_network,
               models_vaeac.prior_network,
               models_vaeac.generative_network)
    SG.eval()
    if cuda:
        SG = SG.cuda(device)

    SG.load_state_dict(torch.load("saves/model_saves/ShapeInpainting_VAEAC_fourier_cnn_o2.pt", map_location=device))

    data = TrainSetIntact(100, crop_dim, n_test=1, test=True)
    loader = DataLoader(data, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    print(f"Generator Number of Parameters: {sum(p.numel() for p in SG.parameters() if p.requires_grad)}")

    DI = DensityInpainting(cuda, device)

    BI_eval = Evaluate(SG, forward, loader, cuda, device, order=order)

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CSE_results = BI_eval.cell_size_error_fourier()
    torch.save(CSE_results, "saves/CSE_VAEAC_mean_results")
    time.time() - start_time

    CSE_results = torch.load("saves/CSE_VAEAC_mean_results")
    BI_eval.cell_size_error_fourier(results=CSE_results, save_as="CSE_VAEAC_mean");

    start_time = time.time()
    with torch.cuda.amp.autocast():
        CEE_results = BI_eval.cell_eccentricity_error_fourier()
    torch.save(CEE_results, "saves/CEE_VAEAC_mean_results")
    time.time() - start_time

    CEE_results = torch.load("saves/CEE_VAEAC_mean_results")
    BI_eval.cell_eccentricity_error_fourier(results=CEE_results, save_as="CEE_VAEAC_mean");

    start_time = time.time()
    with torch.cuda.amp.autocast():
        PCA_results = BI_eval.pca_fourier();
    torch.save(PCA_results, "saves/PCA_VAEAC_mean_results")
    time.time() - start_time

    PCA_results = torch.load("saves/PCA_VAEAC_mean_results")
    BI_eval.pca_fourier(results=PCA_results, save_as="PCA_VAEAC_mean");
