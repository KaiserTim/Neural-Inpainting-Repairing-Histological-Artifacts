import os
import torch
import celldetection as cd

from utils.ops import compute_kde  # This only works if the main package is in the python path
from .models_GatedUNet import GatedUNet
from torchvision.transforms.functional import resize


class DensityInpainting:
    def __init__(self, cuda, device):
        self.cuda = cuda
        self.device = device
        DG = GatedUNet(1, 1, 8, crop_dim=8, dilation=False, condition_channels=1, kernel_size=3, padding=1)  # crop_dim=8 is a hack here
        if cuda:
            DG = DG.cuda(device)
        DG.eval()
        dirname = os.path.dirname(__file__)
        DG.load_state_dict(torch.load(os.path.join(dirname, 
                                                   "model_saves/DensityFill_GatedUNet_bs64_lr2.0e-03_wu8_wd0.0e+00_2.pt"),
                                                   map_location=device))
        self.DG = DG
        
    def count_parameters(self):
        return sum(p.numel() for p in self.DG.parameters() if p.requires_grad)

    def fill_locations(self, cpn, mask, downscale=4, return_img=False):
        """cpn: CPN output dictionary"""

        bs, _, crop_dim, _ = mask.shape
        xy_batch = cpn["xy"].copy()  # list of tensors of shape [n_cells, 2]
        kde = torch.as_tensor(compute_kde(cd.asnumpy(xy_batch), crop_dim, downscale=downscale),
                              dtype=torch.half)  # [bs, 1, crop_dim/downscale, crop_dim/downscale]
        if self.cuda:
            kde = kde.cuda(self.device)
        mask_small = resize(mask, (crop_dim//downscale, crop_dim//downscale)).float()
        kde = (kde - 0.02635) * 18  # Normalize approximately to [-1,1] and 0 mean
        DG_out = self.DG(kde, condition=mask_small, tanh=True)  # Account for network normalization
        DG_painting = DG_out*mask_small.float() + kde*(1-mask_small.float())
        DG_painting = DG_painting / 18 + 0.02635  # Revert the normalization to restore correct probability mass
        
        # Sample new cells
        rand = torch.rand(bs, crop_dim//downscale, crop_dim//downscale)
        if self.cuda:
            rand = rand.cuda(self.device)
        locs_img = rand < DG_painting.squeeze(1)

        if return_img:
            locs_img_large = torch.zeros_like(mask).squeeze(1)
            idx = torch.where(locs_img == 1)
            for i in range(len(idx[0])):
                locs_img_large[idx[0][i], idx[1][i]*downscale, idx[2][i]*downscale] = 1 
            return locs_img_large

        locs_in_mask = []  # List of tensors of shape [n_cells, 2] for cells inside the masked area
        for i in range(bs):
            x, y = torch.where(locs_img[i]*mask_small[i,0] == 1)
            locs_in_mask.append(torch.vstack((y, x)).transpose(0,1)*downscale)  # scale back up
            xy_batch[i] = torch.cat((xy_batch[i], locs_in_mask[i]), dim=0)  # Add the new coordinates

        return xy_batch, locs_in_mask
