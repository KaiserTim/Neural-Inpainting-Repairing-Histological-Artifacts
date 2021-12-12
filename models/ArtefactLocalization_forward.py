import os
import torch

from .models_UNet import UNet


class ArtefactLocalization:
    def __init__(self, cuda, device, crop_dim=256):
        predictor = UNet(1, 1, 16, crop_dim, 1, dilation=False, kernel_size=3, padding=1, bias=False, padding_mode="reflect")
        dirname = os.path.dirname(__file__)
        predictor.load_state_dict(torch.load(os.path.join(dirname, "model_saves/predictor_tuned.pt"), map_location=device))
        predictor.to(device)
        predictor.eval()
        predictor.requires_grad_(False)
        self.predictor = predictor
        self.cuda = cuda
        self.device = device
        
    def forward(self, crop):
        if self.cuda:
            crop = crop.cuda(self.device)
        return self.predictor(crop).sigmoid().round()
