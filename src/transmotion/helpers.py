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
            inpaint_res = nets.inpaint(
                source_img=src_img,
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


def param_name_iou(left_name: str, right_name: str) -> float:
    lhs = set(left_name.split("."))
    rhs = set(right_name.split("."))
    intersection = len(lhs.intersection(rhs))
    union_ = len(lhs.union(rhs))
    return intersection / union_


def reconcile_weights(orig_state_dict: Dict, new_state_dict: Dict):
    # classify params by their shapes (those have to match)
    shapes = {}

    def _add_shape(k, v, model):
        shape = tuple(v.shape)

        if shape not in shapes:
            shapes[shape] = {"old": [], "new": []}

        shapes[shape][model].append(k)

    for k, v in orig_state_dict.items():
        _add_shape(k, v, "old")

    for k, v in new_state_dict.items():
        _add_shape(k, v, "new")

    # more than one param name per shape? now we need to
    conflicts = {}
    mappings = {}
    for k, v in shapes.items():
        h, w = len(v["old"]), len(v["new"])

        if h != w:
            raise ValueError(
                f"Not same number of params with {k} shape: Old: {v['old']} New: {v['new']}"
            )

        if h > 1:
            # find intersectio over union of param names.
            conflicts[k] = np.zeros((h, w), dtype=np.float32)

            for ridx, old_par in enumerate(v["new"]):
                for cidx, new_par in enumerate(v["old"]):
                    conflicts[k][ridx, cidx] = param_name_iou(old_par, new_par)

            # if the best matches are diagonal we can asusme that params are at oreder
            # (we didn't change te order of params in the modules, just some names)
            best_fits = conflicts[k].argmax(axis=1)
            if not (best_fits == np.arange(h)).all():
                # if this condtioin is broken. it might be an issue with the measurement.
                # we compare argmax to digaonal, digaonal must be >= argmax if the order is correct
                if not (
                    conflicts[k][np.arange(h), best_fits] <= np.diagonal(conflicts[k])
                ).all():
                    print(f"{k} shape param match is out of order")
                    continue
        mappings.update(
            {old_name: new_name for old_name, new_name in zip(v["old"], v["new"])}
        )

    return mappings


def import_state_dict(old: Dict, new_: Dict) -> Dict:
    """
    Use heuristics to load weights from original paper to current weights.
    Models haven't changed much from the original, mostly ordered stayed the same.

    **The matching heuristics are as follows**:
    1. Match according to weight shapes (model must have the same shapes and the same number of weight for each shape)
    2. In case more than one weight share shape, perform text similarity on weight name, make sure that the order of old weights matches the order of new weights

    """
    mappings = reconcile_weights(old, new_)
    sdict = {mappings[k]: v for k, v in old.items()}
    return sdict


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
