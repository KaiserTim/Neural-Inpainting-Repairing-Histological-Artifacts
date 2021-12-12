import numpy as np
import torch

from .ArtefactLocalization_forward import ArtefactLocalization
from .BinaryInpainting_forward import BinaryInpainting
from .ImageInpainting_forward import ImageInpainting
from .cpn_custom_forward_2 import CPN


class NeuralInpainting:
    """"""
    def __init__(self, cuda, device, crop_dim=256, order=6, max_len=192):
        assert crop_dim%8 == 0, "crop_dim needs to be divisible by 8."
        self.cuda = cuda
        self.device = device
        self.AL = ArtefactLocalization(cuda, device, crop_dim)
        self.BI = BinaryInpainting(cuda, device, max_len, order)
        self.II = ImageInpainting(cuda, device)
        self.cpn = CPN(cuda, device, order)
        self.max_len = max_len
        
    def inference(self, crop):
        if self.cuda:
            crop = crop.cuda(self.device)
            
        mask = self.AL.forward(crop)
        crop_masked = crop*(1-mask)
        cpn = self.cpn.inference(crop_masked + mask)  # Make the mask "white" to prevent cell detection by CPN
        labels_prd, pop_idx = self.BI.forward(cpn, mask)
        assert len(pop_idx) == 0, f"Some crops contained more than {self.max_len} cells."
        labels_prd = torch.as_tensor(np.stack(labels_prd, axis=0)).unsqueeze(1)  # [bs, 1, crop_dim, crop_dim]
        if self.cuda:
            labels_prd = labels_prd.cuda(self.device)
        
        IG_out = self.II.forward(crop_masked, mask, labels_prd)
        painting = IG_out*mask + crop*(1-mask)
        NIN = {"crop": crop, "mask": mask, "crop_masked": crop_masked, "cpn": cpn, "binary_painting": labels_prd, "painting": painting}
        return NIN
