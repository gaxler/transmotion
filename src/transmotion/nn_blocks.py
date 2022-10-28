"""

.. include:: ../docs/nn_blocks.md
"""
from statistics import mode
from typing import List, Tuple

import torch as th
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(
        self, block_expansion: int, in_dim: int, num_blocks: int = 3, max_features=256
    ):
        """
        :param:
        `block_expansion` - base internal dimension of each block. This base grows by a power of 2 for every block depth
        :param: `num_blocks`
        """
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_dim if i == 0 else min(max_features, block_expansion * (2**i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x: th.Tensor) -> List[th.Tensor]:
        """
        - Input shpae: `[batch, c, height, width]`
        - Output shape: `[ ... [batch, c_i, h_i, w_i] ...]`
        """
        outs = [x]
        # print('encoder:' ,outs[-1].shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
            # print('encoder:' ,outs[-1].shape)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    out_channels: List[int]
    out_dim: int

    def __init__(
        self,
        block_expansion: int,
        in_dim: int,
        num_blocks: int = 3,
        max_features: int = 256,
    ):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:

            block_dim = block_expansion * 2 ** (i + 1)
            in_filters = min(max_features, block_dim)
            if i < num_blocks - 1:
                in_filters *= 2

            # in_filters = (1 if i == num_blocks - 1 else 2) * min(
            #     max_features, block_expansion * (2 ** (i + 1))
            # )
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2**i))
            up_blocks.append(
                UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1)
            )

        self.up_blocks = nn.ModuleList(up_blocks)

        # TODO: this assumes that block_expansion is always smaller than max_features
        self.out_channels.append(block_expansion + in_dim)
        self.out_dim = self.out_channels[-1]
        # self.out_filters = block_expansion + in_features

    def forward(self, x: List[th.Tensor]) -> th.Tensor:
        """
        Take a list of feature maps of shapes `[ ... [b, c_i, h_i, w_i]... ]`.
        Upsample and add skip connections

        TODO:
        - [ ] add network arch sketch

        :param: `x` - output of the src.nn_blocks.Encoder
        """
        out = x.pop()
        outs = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = th.cat([out, skip], dim=1)
            outs.append(out)
        return outs

        # Why do we need this mode thing? always return the full feature list, use the last one if thats what the caller needs.
        # if mode == 0:
        #     return out
        # else:
        #     return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    The input is downsampled with a CNN based encoder and upsampled with a CNN based decoder.
    A bit similar to a U-Net architecure where each upsampled features map has a residual connection to a downsampled feature map from the decoder.
    """

    out_channels: List[int]
    last_out_dim: int

    def __init__(
        self, block_expansion: int, in_features: int, num_blocks=3, max_features=256
    ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)

        self.out_channels = self.decoder.out_channels
        self.last_out_dim = self.decoder.out_dim
        # self.out_filters = self.decoder.out_filters

    def forward(self, x: th.Tensor) -> List[th.Tensor]:
        """
        Encoder a feature map `x` and decode it back.
        :param: `x` - feature map of shape `[b, c, h, w]`

        :return: List[th.Tensor] intermediate decoder feature maps, with increasing spatial dimensions h_i & w_i ande decreaseing dimension c_i
        """

        return self.decoder(self.encoder(x))


class ResBlock2d(nn.Module):
    """
    Residual block that preserve spatial resolution
    """

    def __init__(self, in_dim: int, kernel_size: int, padding: int):
        """"""
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_dim, affine=True)

    def forward(self, x: th.Tensor):
        """
        Take feature map tensor as input

        Expect input of the shape: `[batch, dim, height, width]`
        """
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
    ):
        """"""
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.InstanceNorm2d(out_dim, affine=True)

    def forward(self, x):
        """
        Take feature map tensor as input

        Expect input of the shape: `[batch, dim, height, width]`
        """
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block with `Average Pooling`

    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """
        Take feature map tensor as input

        Expect input of the shape: `[batch, dim, height, width]`
        """
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        groups: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.InstanceNorm2d(out_dim, affine=True)

    def forward(self, x):
        """`[batch, dim, height, width]`"""
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Vgg19(th.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """

    def __init__(
        self, vgg_pretrained_features: th.nn.Module, requires_grad: bool = False
    ):
        super(Vgg19, self).__init__()
        self.slice1 = th.nn.Sequential()
        self.slice2 = th.nn.Sequential()
        self.slice3 = th.nn.Sequential()
        self.slice4 = th.nn.Sequential()
        self.slice5 = th.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = th.nn.Parameter(
            data=th.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)),
            requires_grad=False,
        )
        self.std = th.nn.Parameter(
            data=th.tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)),
            requires_grad=False,
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def get_pretrained_vgg19(requires_grad: bool = False) -> Tuple[th.nn.Module, int]:
    """
    :return: VGG19 module and the length of its output (number of feature maps)
    """
    from torchvision import models

    vgg_pretrained_features = models.vgg19(
        weights=models.VGG19_Weights.IMAGENET1K_V1
    ).features
    return Vgg19(vgg_pretrained_features, requires_grad=requires_grad), 5
