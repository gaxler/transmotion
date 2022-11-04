from cmath import isnan
import dataclasses
import itertools as itt
import re
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch as th

from .dense_motion import (
    BGMotionParam,
    BGMotionPredictor,
    DenseMotionNetwork,
    DenseMotionResult,
)
from .inpainting import InpaintingNetwork, InpaintingResult
from .kp_detection import KPDetector, KPResult
from .weights import import_state_dict
from .configs import TPSConfig, DenseMotionConf, InpaintingConfig

def chain_param(*modules) -> Iterable[th.nn.parameter.Parameter]:
    return itt.chain(*(m.parameters() for m in modules))


@dataclasses.dataclass
class NetworkBundleResult:
    """
    The result of a forward pass. This includes traintime debugging information like source and target ke:50y-points
    Optical flow maps, deformed feature maps etc... Final generated frame is available as a property generated_image
    """

    source_keypoints: KPResult
    driving_keypoints: KPResult
    background_param: BGMotionParam #| None
    dense_motion: DenseMotionResult
    inpainting: InpaintingResult

    @property
    def generated_image(self) -> th.Tensor:
        """
        Final result. Generated source iamge that matches the pose of the drivign image

        .. note:: Detached from autograd on copied to CPU
        """
        return self.inpainting.inpainted_img.detach().cpu()


@dataclasses.dataclass
class NetworksBundle:
    key_points: KPDetector
    dense_motion: DenseMotionNetwork
    inpaint: InpaintingNetwork
    background_motion: BGMotionPredictor #| None

    def eval(self):
        """
        Turn all networks to eval mode and remove the background prediction network.
        Background prediction is not used in prediction
        """
        return NetworksBundle(
            key_points=self.key_points.eval(),
            dense_motion=self.dense_motion.eval(),
            inpaint=self.inpaint.eval(),
            background_motion=None,
        )

    def forward(
        self, src_img: th.Tensor, drv_img: th.Tensor, dropout_prob: float = 0.0
    ) -> NetworkBundleResult:
        kp_src = self.key_points(src_img)
        kp_drv = self.key_points(drv_img)

        bg_param: BGMotionParam
        bg_param = (
            self.background_motion(src_img, drv_img)
            if self.background_motion is not None
            else None
        )

        motion_res: DenseMotionResult
        motion_res = self.dense_motion(
            source_img=src_img,
            src_kp=kp_src,
            drv_kp=kp_drv,
            bg_param=bg_param,
            dropout_prob=dropout_prob,
        )

        inpaint_res: InpaintingResult
        inpaint_res = self.inpaint(
            source_img=src_img,
            occlusion_masks=motion_res.occlusion_masks,
            optical_flow=motion_res.optical_flow,
        )
        return NetworkBundleResult(
            source_keypoints=kp_src,
            driving_keypoints=kp_drv,
            dense_motion=motion_res,
            inpainting=inpaint_res,
            background_param=bg_param,
        )

    def parameters(self) -> Iterable[th.nn.parameter.Parameter]:
        main_params = chain_param(self.key_points, self.dense_motion, self.inpaint)
        if self.background_motion is not None:
            main_params = itt.chain(main_params, self.background_motion.parameters())
        return main_params

    def clear_grads(self):
        for par in self.parameters():
            par.grad = None

    def no_background(self):
        """Copy of Network refernces without the background predictor"""
        return NetworksBundle(
            key_points=self.key_points,
            dense_motion=self.dense_motion,
            inpaint=self.inpaint,
            background_motion=None,
        )

    def load_original_weights(self, fpath: str):
        _name_to_new_module = {
            "kp_detector": self.key_points,
            "dense_motion_network": self.dense_motion,
            "inpainting_network": self.inpaint,
            "bg_predictor": self.background_motion,
        }
        # TODO: Make common device for all nets in bundle 
        saved_weights = th.load(fpath, map_location=th.device("cpu"))

        for name_, sd in saved_weights.items():
            if name_ not in _name_to_new_module:
                continue
            net = _name_to_new_module[name_]
            new_sd = import_state_dict(sd, net.state_dict())
            net.load_state_dict(new_sd)


def get_kp_normalized_forward_func(
    src_img: th.Tensor,
    intial_drv_img: th.Tensor,
    nets: NetworksBundle,
) -> Callable[[th.Tensor], NetworkBundleResult]:

    with th.no_grad():
        kp_src: KPResult
        kp_src = nets.key_points(src_img)
        kp_initial_drv: KPResult
        kp_initial_drv = nets.key_points(intial_drv_img)

        src_inpaint_encodings = nets.inpaint.fwd_encoder(src_img)

    def infer_func(drv_img: th.Tensor) -> NetworkBundleResult:
        """forward func"""
        with th.no_grad():
            kp_drv: KPResult
            kp_drv = nets.key_points(drv_img)
            kp_drv = kp_src.normalize_relative_to(
                init_=kp_initial_drv, cur_=kp_drv
            )

            motion_res: DenseMotionResult
            motion_res = nets.dense_motion(
                source_img=src_img,
                src_kp=kp_src,
                drv_kp=kp_drv,
                bg_param=None,
                dropout_prob=0.0,
            )

            inpaint_res: InpaintingResult
            inpaint_res = nets.inpaint.fwd_decoder(
                source_img=src_img,
                raw_encoding=src_inpaint_encodings,
                occlusion_masks=motion_res.occlusion_masks,
                optical_flow=motion_res.optical_flow,
            )
        return NetworkBundleResult(
            source_keypoints=kp_src,
            driving_keypoints=kp_drv,
            dense_motion=motion_res,
            inpainting=inpaint_res,
            background_param=None,
        )

    return infer_func


def load_original_weights(fpath: str, nets: NetworksBundle) -> NetworksBundle:
    _name_to_new_module = {
        "kp_detector": nets.key_points,
        "dense_motion_network": nets.dense_motion,
        "inpainting_network": nets.inpaint,
        "bg_predictor": nets.background_motion,
    }

    saved_weights = th.load(fpath, map_location=th.device("cpu"))

    for name_, sd in saved_weights.items():
        if name_ not in _name_to_new_module:
            continue
        net = _name_to_new_module[name_]
        new_sd = import_state_dict(sd, net.state_dict())
        net.load_state_dict(new_sd)

    return nets


def build_nets(
    tps: TPSConfig, dense: DenseMotionConf, inpaint: InpaintingConfig
) -> NetworksBundle:
    nets = NetworksBundle(
        key_points=KPDetector(tps),
        dense_motion=DenseMotionNetwork(dense),
        inpaint=InpaintingNetwork(inpaint),
        background_motion=BGMotionPredictor(),
    )

    return nets
