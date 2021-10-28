import itertools as it
import more_itertools as mit

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

# %%
if False:
    # %%
    import os
    import hydra
    from omegaconf import OmegaConf as OC
    PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
    os.chdir(PROJECT_DIR)
    os.listdir()

    # %%
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
    overrides = []
    cfg = hydra.compose(config_name='autoencoder', overrides=overrides)
    print(OC.to_yaml(cfg))

# %%
class ConvBlock(nn.Module):
    '''Reduce spatial resolution by 2'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
        nn.MaxPool2d(2),
        nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class DeConvBlock(nn.Module):
    '''Increase spatial resolution by 2'''
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
        nn.Upsample(scale_factor=2),
        # nn.ReLU(),
        activation,
        )

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = [in_channels]+list(channel_list)
        layers = []
        for c_in, c_out in mit.pairwise(channel_list):
            layers.extend([nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=2), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=2) for c_in, c_out in mit.pairwise(channel_list)])

    def forward(self, x):
        return self.layers(x)

class CNN_BN(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = [in_channels]+list(channel_list)
        layers = []
        for c_in, c_out in mit.pairwise(channel_list):
            layers.extend([nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(c_out), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=2) for c_in, c_out in mit.pairwise(channel_list)])

    def forward(self, x):
        return self.layers(x)

class DCNN(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = list(reversed([in_channels]+list(channel_list)))
        layers = []
        for c_in, c_out in mit.pairwise(channel_list[:-1]):
            layers.extend([nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU()])
        # layers.extend([nn.ConvTranspose2d(in_channels=channel_list[-2], out_channels=channel_list[-1], kernel_size=3, padding=1, output_padding=1, stride=2), nn.Sigmoid()])
        layers.extend([nn.ConvTranspose2d(in_channels=channel_list[-2], out_channels=channel_list[-1], kernel_size=3, padding=1, output_padding=1, stride=2), nn.Identity()])
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, output_padding=1, stride=2) for c_in, c_out in mit.pairwise(reversed(channel_list))])

    def forward(self, x):
        return self.layers(x)

class UpCNN(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = list(reversed([in_channels]+list(channel_list)))
        layers = []
        for c_in, c_out in mit.pairwise(channel_list[:-1]):
            layers.extend([nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.Upsample(scale_factor=2)])
        layers.extend([nn.Conv2d(in_channels=channel_list[-2], out_channels=channel_list[-1], kernel_size=3, padding=1, stride=1), nn.Identity(), nn.Upsample(scale_factor=2)])
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, output_padding=1, stride=2) for c_in, c_out in mit.pairwise(reversed(channel_list))])

    def forward(self, x):
        return self.layers(x)

class UpCNN_BN(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = list(reversed([in_channels]+list(channel_list)))
        layers = []
        for c_in, c_out in mit.pairwise(channel_list[:-1]):
            layers.extend([nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.Upsample(scale_factor=2)])
        # layers.extend([nn.Conv2d(in_channels=channel_list[-2], out_channels=channel_list[-1], kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(channel_list[-1]), nn.Identity(), nn.Upsample(scale_factor=2)])
        layers.extend([nn.Conv2d(in_channels=channel_list[-2], out_channels=channel_list[-1], kernel_size=3, padding=1, stride=1), nn.Identity(), nn.Upsample(scale_factor=2)])
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, output_padding=1, stride=2) for c_in, c_out in mit.pairwise(reversed(channel_list))])

    def forward(self, x):
        return self.layers(x)

class CNN_Blocks(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = [in_channels]+list(channel_list)
        self.layers = nn.Sequential(*[ConvBlock(c_in, c_out) for c_in, c_out in mit.pairwise(channel_list)])

    def forward(self, x):
        return self.layers(x)

class DCNN_Blocks(nn.Module):
    def __init__(self, info, channel_list):
        super().__init__()
        in_channels = info['in_channels']
        channel_list = list(reversed([in_channels]+list(channel_list)))
        layers = [DeConvBlock(c_in, c_out, nn.ReLU()) for c_in, c_out in mit.pairwise(channel_list[:-1])]
        layers.append(DeConvBlock(channel_list[-2], channel_list[-1], nn.Identity()))
        self.layers = nn.Sequential(*layers)
        # self.layers = nn.Sequential(*[DeConvBlock(c_in, c_out) for c_in, c_out in mit.pairwise(reversed(channel_list))])

    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, info, encoder, decoder):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder, info)
        self.decoder = hydra.utils.instantiate(decoder, info)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# %%
if __name__ == '__main__':
    # %%
    x = torch.rand(10,1,64,128)


    layers = nn.Sequential(
    nn.Conv2d(1,8, kernel_size=3, padding=1)
    )
    channel_list=cfg.channel_list
    [nn.Conv2d(1,8, kernel_size=3, padding=1) for c in channel_list]
    l=nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1)
    l=nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, output_padding=1, stride=2)

    x.shape
    l(x).shape


    help(nn.Conv2d)
    layers

    help(hydra.utils.instantiate)
    info={'in_channels': 1}
    m=hydra.utils.instantiate(cfg.model, info)
    m(x).shape

    m.encoder
    layers(x).shape


    ae = AutoEncoder()
