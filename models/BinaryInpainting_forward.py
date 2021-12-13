import os
import numpy as np
import torch
import celldetection as cd
import torch.nn.functional as F

from utils.ops import pad_batch  # This only works if the main package is in the Python path
from .cGlow.CGlowModel import CondGlowModel
from .density_fill_forward import DensityInpainting


class BinaryInpainting:
    def __init__(self, cuda, device, max_len, order=2):
        self.cuda = cuda
        self.device = device
        # Params for cGlow model
        f_dim = 4*order
        x_size = (1,max_len,f_dim)
        y_size = (1,max_len,f_dim)
        x_hidden_channels = 128
        x_hidden_size = 64
        y_hidden_channels = 256
        K = 16
        L = 3
        learn_top = False
        y_bins = 2
        # cGlow model 
        SG = CondGlowModel(x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, K, L, learn_top, y_bins)
        dirname = os.path.dirname(__file__)
        SG.load_state_dict(torch.load(os.path.join(dirname, "model_saves/ShapeInpainting_o2_ml192_bs64_lr5.0e-03_lrdc.pt"),
                                      map_location=device))
        SG.to(device)
        SG.eval()
        self.SG = SG
        
        self.DI = DensityInpainting(cuda, device)
        self.max_len = max_len
        self.f_dim = f_dim
        self.order = order
        
        print(f"Loaded BinaryInpainting Model with {self.count_parameters():,} parameters")
        self.SG.requires_grad_(False)
                
    def count_parameters(self):
        SG_params = sum(p.numel() for p in self.SG.parameters() if p.requires_grad)
        DI_params = self.DI.count_parameters()
        return SG_params + DI_params
        
    def forward(self, cpn, mask, return_labels=False):
        """Expects cpn output dictionary. It needs to correspond to a masked crop."""
        if self.cuda:
            mask = mask.cuda(self.device)
        bs = mask.shape[0]
        with torch.no_grad():
            locs, locs_in_mask = self.DI.fill_locations(cpn, mask)
        n_new_cells = [locs_in_mask[i].shape[0] for i in range(len(locs_in_mask))]
        n_cells = [locs[i].shape[0] for i in range(len(locs))]

        # Get Fourier Tensor 
        fourier_prd = cpn["final_fourier"]
        pop_idx = []   # list of indices with crops with n_cells > max_len        
        for i in range(bs-1,-1,-1):  # range(bs) but backwards
            if n_cells[i] >= self.max_len:
                pop_idx.append(i)
                fourier_prd.pop(i)
                n_cells.pop(i)
                n_new_cells.pop(i)
                locs.pop(i)
                locs_in_mask.pop(i)
            else:
                fourier_prd[i] = fourier_prd[i].reshape(-1, self.f_dim)
                # Prepare Fourier Tensor for inference of new cells
                fourier_prd[i] = F.pad(fourier_prd[i], (0,0,0,n_new_cells[i]))  # [T, f_dim]
        bs -= len(pop_idx)

        fourier_prd, pad_mask = pad_batch(fourier_prd, max_len=self.max_len, f_dim=self.f_dim, cuda=self.cuda, device=self.device)  # [bs, max_len, f_dim]
        fourier_prd = fourier_prd.unsqueeze(1)  # Channel dimension

        loc_mask = torch.zeros_like(fourier_prd, dtype=bool)  # [bs, 1, max_len, f_dim]
        if self.cuda:
            loc_mask = loc_mask.cuda(self.device)
        for i in range(bs):
            loc_mask[i, :, n_cells[i]-n_new_cells[i]:n_cells[i]] = 1  # Make this parallelized? 

        with torch.no_grad():
            sample, _ = self.SG(fourier_prd, y=None, reverse=True)

        fourier_prd[loc_mask] = sample[loc_mask].float()
        segment_prd = []
        for i in range(bs):
            prd = fourier_prd[i, :, :n_cells[i]].reshape(-1, self.order, 4)
            contour = cd.data.cpn.fourier2contour(cd.asnumpy(prd), cd.asnumpy(locs[i]))
            if return_labels:
                segment_prd.append(cd.data.contours2labels(contour, mask.shape[-2:]))  # Keep label indices
            else:
                segment_prd.append(np.any(cd.data.contours2labels(contour, mask.shape[-2:]) > 0, axis=-1))  # Binary cell segmentation
        return segment_prd, pop_idx
