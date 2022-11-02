from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch as th
from einops import rearrange, repeat
from scipy.spatial import ConvexHull
from torch import nn
from torchvision import models


from .configs import TPSConfig

from .cord_warp import make_coordinate_grid


@dataclass
class KPResult:
    """
    The result of a KeyPoint estimation network. With shape of ``[batch, num_tps_transforms, points_per_tps, 2]``
    The network produces parameters for a TPS transformation.

    Each TPS transformation has 5 parameters on every 2D spatial location
    :param foreground_kp: 2D coordinates in ``[-1,1]`` of shape ``[batch, num_tps_transforms, points_per_tps, 2]``
    """

    foreground_kp: th.Tensor
    """2D coordinates in `[-1,1]` of shape `[batch, num_tps_transforms, points_per_tps, 2]`"""
    num_tps: int
    """ Number of Thin-Plate Splines in the predicted key-points """
    pts_per_tps: int
    """ Number of control points per TPS transform"""
    batch_size: int

    def detach_to_cpu(self) -> "KPResult":
        return KPResult(
            foreground_kp=self.foreground_kp.detach().cpu(),
            num_tps=self.num_tps,
            batch_size=self.batch_size,
            pts_per_tps=self.pts_per_tps,
        )

    def to_gaussians(
        self, hw_of_grid: Tuple[int, int], kp_variance: float
    ) -> th.Tensor:  # [bs, num_tps, pts_per_tps, grid_h, grid_w]
        """
        Transform a keypoints of shape ``[batch, num_tps, pts_per_tps, 2]`` into gaussian like representation of shape ``[bs, num_tps, pts_per_tps, grid_h, grid_w]``

        -  By "Gaussian Like" we mean find distance between predicted key-points and a uniform spread grid.
        -  We get back a uniform spread grid with gaussian density centered around each key-point

        We sometimes refer to the number of TPS transforms as K, number of points per TPS transform as N and number of points in a grid as P (= h*w)
        :param hw_of_grid: Height and Width of of the Gaussian representation.

        .. note:: This is a convenience method, this is used when generating heatmaps for the optical-flow estimation
        """
        coordinate_grid = make_coordinate_grid(*hw_of_grid, self.foreground_kp.dtype).to(
            self.foreground_kp.device
        )
        b, k, n, _ = self.foreground_kp.shape
        coord_grid = repeat(coordinate_grid, "h w d -> b k n h w d", b=b, k=k, n=n)
        kps = repeat(self.foreground_kp, "b k n d -> b k n h w d", h=1, w=1)
        # measure distance of each point on the grid fron the predicted key-point.
        # turn this distance into a "gaussian density"

        # TODO: maybe fuse this with torch script?
        gaussians = (coord_grid - kps).pow(2).sum(-1).mul(-0.5).div(kp_variance).exp()
        return gaussians

    def normalize_relative_to(
        self, init_: "KPResult", cur_: "KPResult"
    ) -> "KPResult":
        """
        Normalize key-points relative to some initial key-ponint and current key-points.

        Take the difference vectors of current key-points to some inital key-points. Use the difference vectors to translate current key points.

        :param init_: initial key-points
        :param cur_: current key-points

        .. note:: This normalization is used during inference. We modify the source key-points according to the change to the driving key-points. This way no identity information is leaked from driving to source. 
        """
        drv_diff = cur_.foreground_kp - init_.foreground_kp

        src_for_area = rearrange(
            self.foreground_kp.detach().cpu(), "b k n d -> (b k n) d"
        )
        cur_drv_for_area = rearrange(
            cur_.foreground_kp.detach().cpu(), "b k n d -> (b k n) d"
        )
        src_area = ConvexHull(src_for_area).volume
        cur_drv_area = ConvexHull(cur_drv_for_area).volume
        scale_adaptation = np.sqrt(src_area) / np.sqrt(cur_drv_area)

        scaled_drv_diff = scale_adaptation * drv_diff

        new_fg_kp = self.foreground_kp + scaled_drv_diff

        return KPResult(
            foreground_kp=new_fg_kp,
            num_tps=self.num_tps,
            batch_size=self.batch_size,
            pts_per_tps=self.pts_per_tps,
        )


class KPDetector(nn.Module):
    """
    The network is ResNet-18 that produces **num_tps*points_per_tps*spatial_dim** dimentional feature.
    We late interpert the output as key-points of the shape: ``[num_tps, points_per_tps, spatial_dim]``
    """

    spatial_dim: int = 2
    """ Spatial dimension of key-point coordinates"""

    def __init__(self, cfg: TPSConfig):
        super(KPDetector, self).__init__()
        self.num_tps = cfg.num_tps
        self.points_per_tps = cfg.points_per_tps

        self.fg_encoder = models.resnet18(weights=None)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(
            num_features, self.num_tps * self.points_per_tps * self.spatial_dim
        )

    def forward(self, image: th.Tensor) -> KPResult:
        """
        Runs on a btach of images ``[b, c, h, w]``
        """
        fg_kp = self.fg_encoder(image)
        fg_kp = th.sigmoid(fg_kp).mul(2).sub(1)
        fg_kp = rearrange(
            fg_kp,
            "b (k n d) -> b k n d",
            k=self.num_tps,
            n=self.points_per_tps,
            d=self.spatial_dim,
        )
        return KPResult(
            foreground_kp=fg_kp,
            num_tps=self.num_tps,
            batch_size=fg_kp.shape[0],
            pts_per_tps=self.points_per_tps,
        )


if __name__ == "__main__":

    def kp_shapes(K):
        kpd = KPDetector(K)
        img = th.randn((16, 3, 128, 128))
        res = kpd(img)
        res: KPResult
        assert tuple(res.foreground_kp.shape) == (
            16,
            K,
            kpd.points_per_tps,
            2,
        )  # out: [16, 10, 5, 2]
        assert tuple(res.to_gaussians((32, 32), 0.001).shape) == (
            16,
            K,
            kpd.points_per_tps,
            32,
            32,
        )
        print(f"[{__file__.split('/')[-1]}] Shapes are correct")

    kp_shapes(10)
