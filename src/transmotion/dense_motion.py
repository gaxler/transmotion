"""
"""
from dataclasses import dataclass
import math
from typing import List, Tuple

import torch as th
from torchvision import models
from einops import rearrange, repeat
from torch.nn import functional as F

from .blocks import AntiAliasInterpolation2d

from .kp_detection import KPResult
from .nn_blocks import Hourglass, UpBlock2d
from .cord_warp import (
    ThinPlateSpline,
    make_coordinate_grid,
    conform_to_type_and_device,
    deform_with_5d_deformation,
)

from .configs import DenseMotionConf


@dataclass
class BGMotionParam:
    bg_params: th.Tensor
    """ Affine transform parameters (3x3 shape, with last row as bias [0,0,1]) Shape: ``[bs, 3, 3]`` """


class BGMotionPredictor(th.nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix. The third row is [0 0 1]

    This is a spearate module since we don't train it from the start, it starts as None, and appears as the model has some training weights
    """

    def __init__(self):
        super(BGMotionPredictor, self).__init__()
        self.bg_encoder = models.resnet18(weights=None)
        # inout to BGMotoinPredictor is of 6 dimensions (source and driving concat), so we replace the first conv layer
        self.bg_encoder.conv1 = th.nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = th.nn.Linear(num_features, 6)
        self.bg_encoder.fc.weight.data.zero_()
        # we start from an identity background transformation
        self.bg_encoder.fc.bias.data.copy_(
            th.tensor([1, 0, 0, 0, 1, 0], dtype=th.float)
        )

    def forward(
        self, source_image: th.Tensor, driving_image: th.Tensor
    ) -> BGMotionParam:
        """
        :param source_imgae: [bs, d, h, w]
        :param driving_imgae: [bs, d, h, w]
        """
        bs = source_image.shape[0]
        dtype = source_image.dtype

        out = repeat(th.eye(3, dtype=dtype), "m n -> bs m n", bs=bs)
        # [bs, 2d, h, w]
        inp_ = th.cat([source_image, driving_image], dim=1)

        raw_pred = self.bg_encoder(inp_)
        pred = rearrange(raw_pred, "bs (m n) -> bs m n", m=2, n=3)
        out[:, :2, :] = pred
        return BGMotionParam(bg_params=out)


@dataclass
class WarppedCords:
    """
    :param wrapped_cords: Shape ``[bs, num_tps+1, gird_h, grid_w, 2]``
    """

    warpped_cords: th.Tensor
    num_tps: int

    def deform_frame(self, frame: th.Tensor) -> th.Tensor:
        """
        Multi-deformation. One frames gets deformed by several different deformation coordinates
        Deform image frame according to the wrapped coordinates
        :return: shape ``[b K+1 d h w]``

        """
        return deform_with_5d_deformation(frame, deformation=self.warpped_cords)
        # b, kn_o, h, w, d = self.warpped_cords.shape
        # frame = repeat(frame, "b d h w -> b kn_o d h w", kn_o=kn_o)
        # input_ = rearrange(frame, "b k d h w -> (b k) d h w")
        # cords = rearrange(self.warpped_cords, "b kn_o h w d -> (b kn_o) h w d")
        # deformed_frame = F.grid_sample(input_, cords, align_corners=True)
        # deformed_frame = rearrange(
        #     deformed_frame, "(b kn_o) d h w -> b kn_o d h w", kn_o=kn_o
        # )
        # return deformed_frame


def keypoints_to_heatmap_representation(
    spatial_hw: Tuple[int, int], source: KPResult, driving: KPResult, kp_variance: float
) -> th.Tensor:
    """
    create a heatmap representation from key-point results
    For motion estimation this should be the size of the source images as key-points heatmap is going to be concatenated with tohe source image

    :param: spatial_hw - spatial size of the produces heatmap.
    :source: Predicted Key-points for the source image
    :driving: Predicted Key-points for the driving image
    :kp_variance: Key-points representation is based on a Gaussian distribution. this is the variacne for the distribution we fit.

    :return: Tensor with shpae of `[bs, K*N+1, h, w]
    """
    src_gauss = source.to_gaussians(spatial_hw, kp_variance=kp_variance)
    drv_gauss = driving.to_gaussians(spatial_hw, kp_variance=kp_variance)
    # [bs K N h w] - [bs K N h w]
    hmap = drv_gauss - src_gauss
    bs, num_tps, pts_per_tps, h, w = hmap.shape
    hmap = rearrange(hmap, "b k n h w -> b (k n) h w")
    KN = num_tps * pts_per_tps
    out_hmap = th.zeros((bs, KN + 1, h, w), dtype=hmap.dtype).to(hmap.device)
    out_hmap[:, 1:, :, :] = hmap

    return out_hmap


def tps_warp_to_keypoints_with_background(
    grid_hw: Tuple[int, int],
    source: KPResult,
    driving: KPResult,
    bkg_affine_param: BGMotionParam,  # BGMotionParam | None ƒ
) -> WarppedCords:
    """
    Warp a grid of coordinates of shape ``grid_hw``
    Predicted key-points are for foreground objects. This function warps a grid according to foreground transformation and adds a background
    Each point in the resulting grid says to what point we mapped the current point. Last dimension is the identity, that is the grid without any mapping

    :param bkg_affine_param: Shape `[bs, 3, 3]` if not None

    :return: Tensor with shape `[bs, num_tps+1, gird_h, grid_w, 2]
    """

    h, w = grid_hw
    _cast_to_device = conform_to_type_and_device(source.foreground_kp)

    # orig_tps = OrigTPSComp(mode="kp", bs=driving.batch_size, kp_1=driving.foregroud_kp, kp_2=source.foregroud_kp)
    # fit driving to source spline so destinations is source
    tps = ThinPlateSpline.fit(
        source_pts=driving.foreground_kp, destination_pts=source.foreground_kp
    )

    # shape [h, w, 2]
    identity_grid = make_coordinate_grid(h, w)
    identity_grid = _cast_to_device(identity_grid)
    identity_grid = repeat(identity_grid, "h w d -> b h w d", b=source.batch_size)

    # shape [bs, num_tps, h, w, 2]
    # warp_cords broadcasts the batch_size of keypoints across the batch dim identity_grid
    driving_to_source = tps.warp_cords(identity_grid)
    identity_grid = repeat(identity_grid, "b h w d -> b k h w d", k=1)

    if bkg_affine_param is not None:
        b, k, h, w, d = identity_grid.shape

        # (1) switch to homogenous coordinates
        homogen_grid = _cast_to_device(th.ones((b, k, h, w, d + 1)))
        homogen_grid[:, :, :, :, :d] = identity_grid
        # (2) do the affine transform
        affine = repeat(
            bkg_affine_param.bg_params, "b n m -> b k h w n m", k=1, h=1, w=1
        )
        transformed_grid = th.einsum("bkhwD,bkhwvD->bkhwv", homogen_grid, affine)

        # (3) back from homogenous coordinates
        identity_grid = transformed_grid[..., :2].div(transformed_grid[..., 2:])

    # shape: [bs, num_tps*pts_per_tps+1, h, w, 2]
    warped_kp = th.cat((identity_grid, driving_to_source), dim=1)
    return WarppedCords(warpped_cords=warped_kp, num_tps=source.num_tps)


def tps_dropout_softmax(fmap: th.Tensor, drop_prob: float) -> th.Tensor:
    """
    We want to do some drop out on the K TPS transforms so that the model will learn to use all of them.
    We never drop the background features
    :param fmap: represents features of background and K TPS transforms
    :type fmap: [bs, K+1, h. w]
    """
    # TODO: JIT this (do i need to mark drop_prob as static?)

    # TODO: Optimization?: make this a class that has a device buffer
    should_drop = (th.rand(fmap.shape[0], fmap.shape[1], 1, 1) < drop_prob).to(
        fmap.device
    )

    # never drop the background features (always the first dim)
    should_drop[:, 0, ...] = False

    fmap_max = fmap.max(1, keepdim=True).values
    fmap = fmap.sub(fmap_max).exp()
    # keep mean of activation about the same after dropout
    fmap[:, 1:, ...] = fmap[:, 1:, ...].div(1 - drop_prob)
    fmap = fmap.masked_fill(should_drop, 0)
    fmap = fmap.div(fmap.sum(1, keepdim=True) + 1e06)
    return fmap


class LearnedUpsample(th.nn.Module):
    """
    :param upsampled_dims: Dimension sizes of the upsampled feature maps
    """

    def __init__(self, in_dim: int, num_upsamples: int) -> None:
        super().__init__()
        dims = [in_dim]
        for _ in range(num_upsamples):
            dims.append(dims[-1] // 2)

        in_and_out_dims = zip(dims[:-1], dims[1:])

        self.upsamples = th.nn.ModuleList(
            [
                UpBlock2d(in_, out_, kernel_size=3, padding=1)
                for (in_, out_) in in_and_out_dims
            ]
        )

        self.num_upsamples = num_upsamples
        self.upsampled_dims = dims[1:]

    def forward(self, fmap: th.Tensor) -> List[th.Tensor]:
        """
        :param fmap: Feature map that is going to be upsampled (usually the last one)
        :type fmap: [bs, d, h, w]
        :return: List of upsampled featute maps
        :rtype: [[bs, d // 2, h*2, w*2], [bs, d//4, h*4, w*4], ... ]
        """
        out = [fmap]
        for f in self.upsamples:
            x = out[-1]
            out.append(f(x))
        # don't want to return the input, just the upsampled feature maps
        return out[1:]


class OcclusionMasks(th.nn.Module):
    """
    :param mask_predicotrs: Conv layers followed by Sigmoid activation. kernel_size=7 and padding preserves spatial dimension
    """

    def __init__(self, fmap_channels: List[int]) -> None:
        """
        :param fmap_channels: Number of channels (dimensions) in the feature maps that are going to be used as inputs to occlusion mask predicitons
        """
        super().__init__()

        self.mask_predictors = [
            th.nn.Sequential(
                th.nn.Conv2d(ch, 1, kernel_size=(7, 7), padding=(3, 3)), th.nn.Sigmoid()
            )
            for ch in fmap_channels
        ]

        self.mask_predictors = th.nn.ModuleList(self.mask_predictors)

    def forward(self, fmaps: List[th.Tensor]) -> List[th.Tensor]:
        return [f(fmap) for f, fmap in zip(self.mask_predictors, fmaps)]


@dataclass
class DenseMotionResult:
    """
    Output of the DenseMotionNetwork

    :param deformed_source:  Source image deformed by K TPS transformations (not by the optical flow!)
    :param contribution_maps: Convex weights on each of the TPS transforms & background transform
    :param optical_flow: a grid of coordinates :math:`\in [0,1]`, every coordinate tells us where grid pixel should be moved
    """

    deformed_source: th.Tensor
    """ Shape - ``[bs, K+1, h w]``"""
    contribution_maps: th.Tensor
    """ Shape  - ``[bs, K+1, h, w]``"""
    optical_flow: th.Tensor
    """ Shape - ``[bs, h, w, d (=2)]``"""
    occlusion_masks: List[th.Tensor]
    """ Shpae of element i - ``[bs, d_i, h_i, w_i]``"""


def _raise_error_if_too_many_occlusions(requested, found, upsamps):
    # in case we don't have enough feature maps, lets raise and error
    if len(found) < requested:
        fmaps = found - upsamps

        _msg = f"{requested} Occlusion masks requested, found {fmaps} HourGlass feature maps and {upsamps} UpSampled feature maps"
        raise ValueError(_msg)


dbg_state_dicts = []


def _comp_sd(sd1, sd2):
    res = {}
    for k in sd1:
        v, y = sd1[k], sd2[k]
        norm_diff = th.abs(v - y).sum() / (v.abs().sum() + 1e-9)
        if norm_diff > 0:
            res[k] = norm_diff
    return res


class DenseMotionNetwork(th.nn.Module):
    """
    1. Estimate optical flow from key-point transformatiopn,
    2. Estimate multi-resolution occlusion masks

    :param hg_to_num_mappings: Conv layer that translates HourGlass output dimension to the number of mappings (in this case num of TPS transforms + bkg transform)
    """

    hg_to_num_mappings: th.nn.Module

    def __init__(self, cfg: DenseMotionConf) -> None:
        super().__init__()

        self.downsample = AntiAliasInterpolation2d(
            dim=cfg.in_features, scale=cfg.scale_factor
        )
        K, N = cfg.tps.num_tps, cfg.tps.points_per_tps

        # we concatentae K+1 deformed frames and KN+1 heatmaps
        _in_features = cfg.in_features * (K + 1) + (K * N + 1)
        self.hourglass = Hourglass(
            block_expansion=cfg.base_dim,
            in_features=_in_features,
            num_blocks=cfg.num_blocks,
            max_features=cfg.max_features,
        )

        hg_last_dim = self.hourglass.last_out_dim
        hg_out_dims = self.hourglass.out_channels
        self.hg_to_num_mappings = th.nn.Conv2d(
            in_channels=hg_last_dim,
            out_channels=K + 1,
            kernel_size=(7, 7),
            padding=(3, 3),
        )

        inv_scale_factor_nearest_positive_power_of_2 = max(
            0, int(-math.log(cfg.scale_factor, 2))
        )
        num_upsamples = (
            inv_scale_factor_nearest_positive_power_of_2
            if cfg.num_occlusion_masks > 1
            else 0
        )

        self.upsamples = LearnedUpsample(
            num_upsamples=num_upsamples,
            in_dim=hg_last_dim,
        )

        # we use all the upsampled feature maps as input to occlusion masks production
        num_fmap_to_occlusion_masks = cfg.num_occlusion_masks - num_upsamples

        # TODO:there might be a bug in here. hg_out_dims include the input dim. but the input is not included in the out features
        # TODO: find out if it is a bug, best way to fix it
        # take the hourglass feature maps
        occlusion_input_dims = hg_out_dims[-num_fmap_to_occlusion_masks:]
        # add the upsampled feature maps to them
        occlusion_input_dims += self.upsamples.upsampled_dims

        # in case we don't have enough feature maps, lets raise and error
        _raise_error_if_too_many_occlusions(
            requested=cfg.num_occlusion_masks,
            found=occlusion_input_dims,
            upsamps=num_upsamples,
        )

        self.occlusion_mask_model = OcclusionMasks(fmap_channels=occlusion_input_dims)
        self.num_occlusion_masks = cfg.num_occlusion_masks
        self.kp_variance = cfg.kp_variance

    def infer_optical_flow(
        self, fmap: th.Tensor, warpped_cords: th.Tensor, dropout_prob: float
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Turn the output of the hour-glass architecture into a softmax weights tensor and use that reduce K+1 transformed cords to a single optical-flow tensor.


        :param fmap: Last output of the HourGlass architecture used to infer contribution maps
        :param warpped_cords: Shape: [b k h w d] coordinates after beign warped with a TPS transform
        :type fmap: 4-D Batched Tensor

        :return: Optical-flow Tensor (shape: [b h w d]) and the weights (shape: [b k h w]) assigned to each of the transformations.
        """

        # [bs, K+1, h, w]
        contribution_maps = self.hg_to_num_mappings(fmap)

        if dropout_prob > 0:
            contribution_maps = tps_dropout_softmax(
                contribution_maps, drop_prob=dropout_prob
            )
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)

        # Unify all K TPS transforms into a single coordinate deformation
        # [b, K+1, h w,2]*[b, K+1, h, w, 1]

        # here we do a strange dimension shuffeling to make the broadcasted multiplication memory efficient
        # we do multiple uses of the hw blocks.
        # we gonna load hw from the contrib map and use it twice, once for each dimension of warpped cords
        _contrib_maps = repeat(contribution_maps, "b k h w -> b k d h w", d=1)
        _warpped_cords = rearrange(warpped_cords, "b k h w d -> b k d h w")
        _weighted_cords = _warpped_cords * _contrib_maps
        optical_flow = th.einsum("bKdhw->bhwd", _weighted_cords)

        return optical_flow, contribution_maps

    def maybe_upsample_infer_occlusion_masks(
        self, feature_maps: List[th.Tensor]
    ) -> List[th.Tensor]:

        upsampled_fmaps = self.upsamples(feature_maps[-1])
        num_additional_fmaps = self.num_occlusion_masks - len(upsampled_fmaps)

        input_fmaps_for_occlusion = (
            feature_maps[-num_additional_fmaps:] + upsampled_fmaps
        )
        return self.occlusion_mask_model(input_fmaps_for_occlusion)

    def forward(
        self,
        source_img: th.Tensor,
        src_kp: KPResult,
        drv_kp: KPResult,
        bg_param: BGMotionParam = None,  # | None = None,
        dropout_prob: float = 0.0,
    ) -> DenseMotionResult:
        """
        Flow of the forward pass:

        -  Downsample the source image
        -  Generate heatmap representation out of the source and driving key-points
        -  Fit Thin-Plate Spline to source and driving key-points
        -  Deform source image with TPS transform (happens in :func:`tps_warp_to_keypoints_with_background`)
        -  Concat heatmaps and deformed source and pass that thorough an hour-glass neural-net (HG)
        -  Infer optical-flow from last HG feature-map
        -  Infer occlusion masks from the last HG feature-map
        """

        source_img = self.downsample(source_img)

        _, _, h, w = source_img.shape
        # shape [bs, KN+1, h, w]
        hmap = keypoints_to_heatmap_representation(
            (h, w), source=src_kp, driving=drv_kp, kp_variance=self.kp_variance
        )

        warrped_cords_to_kp = tps_warp_to_keypoints_with_background(
            (h, w), source=src_kp, driving=drv_kp, bkg_affine_param=bg_param
        )
        # [bs, K+1 d h w]
        deformed_source = warrped_cords_to_kp.deform_frame(source_img)

        deformed_inp = rearrange(deformed_source, "b k d h w -> b (k d) h w")
        # [bs, KN+1+(K+1)*d, h, w]
        inp_ = th.cat((hmap, deformed_inp), dim=1)

        all_features_maps = self.hourglass(inp_)

        hg_output = all_features_maps[-1]

        optical_flow, contribution_maps = self.infer_optical_flow(
            fmap=hg_output,
            warpped_cords=warrped_cords_to_kp.warpped_cords,
            dropout_prob=dropout_prob,
        )

        occlusion_masks = self.maybe_upsample_infer_occlusion_masks(all_features_maps)

        return DenseMotionResult(
            deformed_source=deformed_source,
            contribution_maps=contribution_maps,
            optical_flow=optical_flow,
            occlusion_masks=occlusion_masks,
        )


if __name__ == "__main__":
    x = th.arange(10, dtype=th.float32)
    y = th.arange(1, 11, dtype=th.float32)

    print(x.div_(y.mul_(2)))
