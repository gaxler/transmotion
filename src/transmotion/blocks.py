"""
Building Block with trainable parameters
"""
from bisect import bisect_left
from re import I
from typing import List, Sequence, Tuple
from einops import rearrange, repeat
import torch as th
from torch.nn import functional as F


class AntiAliasInterpolation2d(th.nn.Module):
    """
    You can get more info on anti-aliasing for image resize here:
    - [The dangers behind image resizing](https://blog.zuru.tech/machine-learning/2021/08/09/the-dangers-behind-image-resizing)
    """

    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.scale = scale

        if scale == 1.0:
            return

        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1

        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel = _make_gaussian_kernel(
            kernel_size, sigma
        )  # shape [kernel_szie, kernel_size]
        kernel = repeat(kernel, "h w -> d m h w", d=dim, m=1).clone()

        self.register_buffer("weight", kernel)
        self.groups = dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Apply Gaussian Kerenl Before doing nearest neighbor interpolation
        """

        # this is an identity if scale =1.0
        if self.scale == 1.0:
            return x

        # this pads the last two dimensions of X
        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))
        return out


def _make_gaussian_kernel(size: int, std: float) -> th.Tensor:
    k_t = th.arange(size, dtype=th.float32).repeat(size, 1)
    mu = (size - 1) / 2
    ker = th.exp(-(k_t - mu).pow(2).div(2 * std**2))
    kernel = ker * ker.t()

    kernel = kernel / kernel.sum()

    return kernel


class ImagePyramid(th.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """

    def __init__(self, scales: Sequence[float], in_dim: int):
        super(ImagePyramid, self).__init__()

        # make sure we go from small to large images
        self.scales = sorted(scales)
        downs = [AntiAliasInterpolation2d(in_dim, scale) for scale in self.scales]
        self.downs = th.nn.ModuleList(downs)
        # downs = {}
        # for scale in scales:
        #     downs[str(scale).replace(".", "-")] = AntiAliasInterpolation2d(
        #         in_dim, scale
        #     )
        # self.downs = th.nn.ModuleDict(downs)

    def nearest_scale(self, scale: float) -> Tuple[float, th.nn.Module]:
        idx = bisect_left(self.scales, scale)
        return self.scales[idx], self.downs[idx]

    def forward(self, fmap: th.Tensor) -> List[th.Tensor]:
        return [f(fmap) for f in self.downs]
        # out_dict = {}

        # for scale, down_module in self.downs.items():
        #     out_dict["prediction_" + str(scale).replace("-", ".")] = down_module(x)
        # return out_dict


if __name__ == "__main__":
    pass
