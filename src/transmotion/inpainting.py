"""
In-painting network
"""

from dataclasses import dataclass
from re import I
from typing import List
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat, parse_shape

from .configs import InpaintingConfig
from .nn_blocks import DownBlock2d, ResBlock2d, SameBlock2d, UpBlock2d


@dataclass
class InpaintingResult:
    """
    :param deformed_src_fmaps: encoder feture map after deformation and occlusion, those have **gradients on the optical flow tensor**.
    :param deformed_source: Source image **deformed with optical flow**
    :param inpainted_img: generated image from source such that it matches driving pose
    """

    deformed_src_fmaps: List[th.Tensor]
    """ ``[bs, d_i, h_i, w_i]`` i-th element of the list """
    deformed_source: th.Tensor
    """ ``[bs, d, h, w]`` """
    inpainted_img: th.Tensor
    """ ``[bs, d, h, w]`` """


def resize_deform(frame: th.Tensor, deformation: th.Tensor) -> th.Tensor:
    """
    Deform frame according to coordinate deformation.

    :param deformation: A grid of coordinates :math:`\in [0,1]`.
    :type deformation: ``[bs, h1, w1, d (=2)]``
    :param frame: An image of a feature map
    :type frame: ``[bs, d, h2 ,w2]``

    :return: Deformed feature map (``[bs, d, h2 ,w2]``)
    """
    # validate input shapes
    fshape = parse_shape(frame, "bs d h w")
    dshape = parse_shape(deformation, "bs h w d")

    deform = deformation
    # if shapes don;t match, resize
    if any((fshape[k] != dshape[k] for k in "hw")):
        deform = rearrange(deformation, "b h w d -> b d h w")
        deform = F.interpolate(
            deform, size=(fshape["h"], fshape["w"]), mode="bilinear", align_corners=True
        )
        deform = rearrange(deform, "b d h w -> b h w d")

    return F.grid_sample(frame, deform, align_corners=True)


def resize_occlude(frame: th.Tensor, occlusion_mask: th.Tensor) -> th.Tensor:
    """
    Occlude feature map :fmap: using occlusion_mask.
    If mask is not the same size as feature map, use bilinear interpolation on the mask
    to align the sizes.

    :param frame: [bs, d, h1, w1]
    :param occlusion_mask: [bs, 1, h2, w2]

    :return: Occluded image ``[bs, d, h1, w1]``
    """

    fshape = parse_shape(frame, "bs d h w")
    oshape = parse_shape(occlusion_mask, "bs d h w")

    # if shapes don;t match, resize
    if any((fshape[k] != oshape[k] for k in "hw")):
        occlusion_mask = F.interpolate(
            occlusion_mask,
            size=(fshape["h"], fshape["w"]),
            mode="bilinear",
            align_corners=True,
        )

    return frame * occlusion_mask


class InpaintingNetwork(th.nn.Module):
    def __init__(self, cfg: InpaintingConfig) -> None:
        super().__init__()

        self.first = SameBlock2d(
            cfg.in_features, cfg.base_dim, kernel_size=(7, 7), padding=(3, 3)
        )

        down_blocks = []
        up_blocks = []
        res_blocks = []
        for i in range(cfg.num_down_blocks):
            in_dim = min(cfg.max_features, cfg.base_dim * 2**i)
            out_dim = min(cfg.max_features, cfg.base_dim * 2 ** (i + 1))
            down_blocks.append(
                DownBlock2d(
                    in_features=in_dim,
                    out_features=out_dim,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                )
            )

            decoder_in_feature = out_dim
            # we concat regular feature map with deformed and occluded feature maps
            if i < cfg.num_down_blocks - 1:
                decoder_in_feature = 2 * out_dim

            up_blocks.append(
                UpBlock2d(
                    decoder_in_feature, in_dim, kernel_size=(3, 3), padding=(1, 1)
                )
            )
            _resblock = lambda: ResBlock2d(
                decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)
            )
            res_blocks += [_resblock(), _resblock()]

        self.down_blocks = th.nn.ModuleList(down_blocks)
        self.up_blocks = th.nn.ModuleList(up_blocks[::-1])
        self.res_blocks = th.nn.ModuleList(res_blocks[::-1])

        self.final = th.nn.Conv2d(
            cfg.base_dim, cfg.in_features, kernel_size=(7, 7), padding=(3, 3)
        )
        self.num_channels = cfg.in_features

    def forward(
        self,
        source_img: th.Tensor,
        occlusion_masks: List[th.Tensor],
        optical_flow: th.Tensor,
    ) -> InpaintingResult:
        """
        Forward pass has the following flow:

        -  encode source image to produce encoder features
        -  produce optical-flow deformed (:func:`resize_deform`) and mask occluded features (:func:`resize_occlude`). we have two version of the encoder features, one with full gradients and another with gradient only on the optical-flow.
        -  use the deformed features as inputs to the decoder
        -  deform and occlude the source image 
        -  final prediction is a convex combination of the deforemed-occluded source image and the inverse occluded decoder output 


        :param source_img: Source image batch
        :type source_img: ``[bs, 3, h, w]``
        :param occlusion_masks: List of multi scale occlusion masks produced by :class:`transmotion.dense_motion.DenseMotionNetwork`.
        :type occlusion_masks: ``[bs, d_i, h_i, w_i]`` for i-th element of the list
        :param optical_flow: a grid of coordinates, every coordinate tells us where grid pixel should be moved
        :type optical_flow: ``[bs, h, w, d (=2)]`` 


        .. note:: A Check that the number of occlusion masks is consistent with the number of down-blocks happens during config initx
        """

        out = self.first(source_img)
        encoding = [out]
        for f in self.down_blocks:
            encoding.append(f(encoding[-1]))

        out = encoding.pop()
        encoding = encoding[::-1]

        # this one trains the optical flow
        out_ij = resize_deform(out.detach(), optical_flow)
        # occlusion masks infered on deformed images
        out_ij = resize_occlude(out_ij, occlusion_masks[0].detach())

        # this one trains both optical flow and in-painting
        out = resize_deform(out, optical_flow)
        out = resize_occlude(out, occlusion_masks[0])

        warp_encoding = [out_ij]

        # we might get only a single occlusion mask, this is OK.
        # we will broadcast the mask for each feature
        if len(occlusion_masks) == 1:
            occlusion_masks = occlusion_masks * (1 + len(self.up_blocks))

        last_encoding_idx = len(encoding) - 1
        for (i, (encode_i, occlusion_mask)) in enumerate(
            zip(encoding, occlusion_masks[1:])
        ):
            out = self.res_blocks[2 * i](out)
            out = self.res_blocks[2 * i + 1](out)
            out = self.up_blocks[i](out)

            # train the optical flow
            encode_ij = resize_deform(encode_i.detach(), optical_flow)
            encode_ij = resize_occlude(encode_ij, occlusion_mask.detach())
            warp_encoding.append(encode_ij)

            out_encode_i = resize_deform(encode_i, optical_flow)
            out_encode_i = resize_occlude(out_encode_i, occlusion_mask)
            # last encoding is the origianl input, so no concat with up block
            if i < last_encoding_idx:
                out = th.cat([out, out_encode_i], dim=1)

        occlusion_last = occlusion_masks[-1]
        inv_occlusion = 1 - occlusion_last
        out = resize_occlude(out, inv_occlusion) + out_encode_i
        out = self.final(out)
        out = th.sigmoid(out)
        optical_flow_deformed_source = resize_deform(source_img, optical_flow)
        inpainted_img = (
            out * inv_occlusion + optical_flow_deformed_source * occlusion_last
        )

        return InpaintingResult(
            deformed_src_fmaps=warp_encoding,
            deformed_source=optical_flow_deformed_source,
            inpainted_img=inpainted_img,
        )

    def get_encode_no_grad(
        self, driver_img: th.Tensor, occlusion_masks: List[th.Tensor]
    ) -> List[th.Tensor]:
        """
        :return: Decreasing dimension and increasing spatial size
        """
        encodings = []

        def all_encoder_layers():
            """Concat first layer with module list of down blocks"""
            yield self.first
            for down_block in self.down_blocks:
                yield down_block

        with th.no_grad():
            out = driver_img
            rev_occl_masks = occlusion_masks[::-1]
            for (enc_block, mask) in zip(all_encoder_layers(), rev_occl_masks):
                out = enc_block(out)
                occluded_enc = resize_occlude(out, mask)
                encodings.append(occluded_enc)

        return encodings[::-1]
