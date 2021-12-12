import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """PyTorch UNet class
    Init Arguments:
        in_channels/crop_dim: Input dimension needs to be [bs, in_channels+condition_channels, crop_dim, crop_dim]
        width_unit: Number of channels through the layers, ranging from width_unit to 8*width_unit
        condition_channels: Number of channels used for conditional inputs in the forward pass
        blocks: Number of conv-batchnorm-actv blocks in each UNet block
        leaky: Whether to use leakyReLU instead of ReLU
        dilation: Whether to add a series of 2-/4-/8-dilated convolutions in the bottleneck
        batch_norm: Whether to use batch normalization
        sn: Whether to use spectral normalization
        onehot: Whether to one-hot encode the input in the forward pass
        **kwargs: Optional parameters for the convolutional layers
    """
    def __init__(self, in_channels, out_channels, width_unit, crop_dim, blocks=1, leaky=False, dilation=False, 
                 condition_channels=None, batch_norm=True, sn=False, onehot=False, **kwargs):
        super().__init__()

        assert crop_dim % 8 == 0, "The crop dimension has to be divisible by 8."
        self.leaky = leaky
        self.condition_channels = condition_channels
        self.onehot = onehot
        if condition_channels is not None:
            in_channels += condition_channels

        self.tail = nn.Sequential(
            Conv2dBlock(in_channels, width_unit, batch_norm=batch_norm, leaky=leaky, sn=sn, **kwargs),
            ResBlock(width_unit, blocks=blocks, leaky=leaky, batch_norm=batch_norm, sn=sn, **kwargs))

        self.downblock1 = UNetBlock(width_unit, width_unit*2, "down", blocks, leaky, batch_norm=batch_norm, sn=sn, **kwargs)
        self.downblock2 = UNetBlock(width_unit*2, width_unit*4, "down", blocks, leaky, batch_norm=batch_norm, sn=sn, **kwargs)
        self.downblock3 = UNetBlock(width_unit*4, width_unit*8, "down", blocks, leaky, batch_norm=batch_norm, sn=sn, **kwargs)
        if dilation:
            self.floor = nn.Sequential(Conv2dBlock(width_unit*8, width_unit*8, leaky=leaky, batch_norm=batch_norm, sn=sn,
                                                   kernel_size=3, dilation=2, padding=2, padding_mode="reflect"),
                                       Conv2dBlock(width_unit*8, width_unit*8, leaky=leaky, batch_norm=batch_norm, sn=sn,
                                                   kernel_size=3, dilation=4, padding=4, padding_mode="reflect"),
                                       Conv2dBlock(width_unit*8, width_unit*8, leaky=leaky, batch_norm=batch_norm, sn=sn,
                                                   kernel_size=3, dilation=8, padding=8, padding_mode="reflect"),
                                       nn.Upsample(scale_factor=2), nn.Conv2d(width_unit*8, width_unit*4, kernel_size=1))
        else:  
            self.floor = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(width_unit*8, width_unit*4, kernel_size=1))
       
        self.upblock2 = UNetBlock(width_unit*8, width_unit*2, "up", blocks, leaky, batch_norm=batch_norm, sn=sn, **kwargs)
        self.upblock3 = UNetBlock(width_unit*4, width_unit, "up", blocks, leaky, batch_norm=batch_norm, sn=sn, **kwargs)
        self.head = nn.Sequential(Conv2dBlock(width_unit*2, width_unit, leaky=leaky, batch_norm=batch_norm, 
                                              sn=sn, **kwargs),
                                  ResBlock(width_unit, blocks=blocks, leaky=leaky, batch_norm=batch_norm, sn=sn, **kwargs),
                                  nn.Conv2d(width_unit, out_channels, kernel_size=1))

    def forward(self, x, condition=None, sigmoid=False, tanh=False):
        """"""
        if self.onehot:
            x = F.one_hot(x, 3).squeeze(1).float().permute(0,3,1,2)
        if self.condition_channels is not None:
            assert condition is not None, "If condition_channels != None, condition can not be None"
            condition = condition.float()  # [bs, n_cond, crop_dim, crop_dim]
            x = torch.cat((x, condition), dim=1)  # [bs, n_stack+cond, crop_dim, crop_dim]

        skip1 = self.tail(x)                                   # [bs, width_unit  , crop_dim,   crop_dim]
        skip2 = self.downblock1(skip1)                         # [bs, width_unit*2, crop_dim/2, crop_dim/2]
        skip3 = self.downblock2(skip2)                         # [bs, width_unit*4, crop_dim/4, crop_dim/4]
        out = self.downblock3(skip3)                           # [bs, width_unit*8, crop_dim/8, crop_dim/8]
        floor = self.floor(out)                                # [bs, width_unit*4, crop_dim/4, crop_dim/4]
        out = self.upblock2(torch.cat((skip3, floor), dim=1))  # [bs, width_unit*2, crop_dim/2, crop_dim/2]
        out = self.upblock3(torch.cat((skip2, out), dim=1))    # [bs, width_unit,   crop_dim,   crop_dim]
        out = self.head(torch.cat((skip1, out), dim=1))        # [bs, 1,            crop_dim,   crop_dim]
        if sigmoid:
            out = F.sigmoid(out)
        if tanh:
            out = F.tanh(out)
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
class UNetBlock(nn.Module):
    """UNet Block including a ResidualBlock with blocks layers.
    The sampling paramter determines whether to upsample or downsample the
    feature dimensions by a factor of 2."""
    def __init__(self, in_channels, out_channels, sampling, blocks=1, leaky=False, batch_norm=True, sn=False, **kwargs):
        super().__init__()
        assert sampling == "down" or sampling == "up", "Please specify sampling='down' or sampling='up'"

        if sampling == "down":
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
                Conv2dBlock(in_channels, out_channels, batch_norm=batch_norm, leaky=leaky, sn=sn, **kwargs),
                ResBlock(out_channels, blocks=blocks, leaky=leaky, batch_norm=batch_norm, sn=sn, **kwargs))
        else:
            self.block = nn.Sequential(
                Conv2dBlock(in_channels, out_channels*2, batch_norm=batch_norm, leaky=leaky, sn=sn, **kwargs), 
                ResBlock(out_channels*2, blocks=blocks, leaky=leaky, batch_norm=batch_norm, sn=sn, **kwargs),               
                nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(out_channels*2, out_channels, kernel_size=1)))

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with activation, optional BatchNorm2d and LeakyReLU."""
    def __init__(self, channels, blocks=1, leaky=False, batch_norm=True, sn=False, **kwargs):
        super().__init__()

        layers = []
        for _ in range(blocks - 1):
            layers.append(Conv2dBlock(channels, channels, batch_norm=batch_norm, leaky=leaky, sn=sn, **kwargs))

        self.block = nn.Sequential(*layers)
        self.head = Conv2dBlock(channels, channels, batch_norm=False, leaky=None, sn=sn, **kwargs)
        if batch_norm:
            self.actv = nn.Sequential(nn.BatchNorm2d(channels), 
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True))
        else:
            self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.head(out)
        out += x
        out = self.actv(out)
        return out
    

class Conv2dBlock(nn.Module):
    """Convolutional layer with activation, optional BatchNorm2d and LeakyReLU instead of ReLU."""
    def __init__(self, in_channels, out_channels, batch_norm=True, leaky=False, sn=False, **kwargs):
        super().__init__()
        self.batch_norm = batch_norm
        self.leaky = leaky
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        if sn:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
        if batch_norm:
            self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feature = self.conv2d(x)
        if self.batch_norm:
            feature = self.batch_norm2d(feature)
        if self.leaky is not None:
            return self.activation(feature)
        else:
            return feature               
