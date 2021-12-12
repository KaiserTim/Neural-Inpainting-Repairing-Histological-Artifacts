import os
import torch

from .models_GatedUNet import GatedUNet
from utils.ops import get_noise  # This only works if the main package is in the python path


class ImageInpainting:
    """"""
    def __init__(self, cuda, device):
        self.cuda = cuda
        self.device = device
        crop_dim = 8  # Hack
        IG = GatedUNet(1, 1, 16, crop_dim, dilation=True, condition_channels=3, kernel_size=3, padding=1, padding_mode="reflect")
        dirname = os.path.dirname(__file__)
        IG.load_state_dict(torch.load(os.path.join(dirname, "model_saves/G_GatedUNet_UNet_4_3_2.pt"), map_location=device))
        IG.to(device)
        IG.eval()
        self.IG = IG
        
        print(f"Loaded ImageInpainting Model with {self.count_parameters():,} parameters")
        self.IG.requires_grad_(False)
                
    def count_parameters(self):
        return sum(p.numel() for p in self.IG.parameters() if p.requires_grad)
        
    def forward(self, crop, mask, condition):
        # Asserts
        condition = torch.as_tensor(condition)  # Still fast if condition is already a tensor
        if self.cuda:
            crop = crop.cuda(self.device)
            mask = mask.cuda(self.device)
            condition = condition.cuda(self.device)
        
        if len(condition.shape) == 2:
            condition = condition.unsqueeze(0)
        if len(condition.shape) == 3:
            condition = condition.unsqueeze(1)
            
        bs, _, crop_dim, _ = crop.shape
        noise = get_noise(0, 1, crop_dim, bs)
        if self.cuda:
            noise = noise.cuda(self.device)
        IG_out = self.IG(crop*(1-mask), condition=torch.cat((mask, condition, noise), dim=1), tanh=True)
        IG_painting = IG_out*mask + crop*(1-mask)
        return IG_painting
