import os
import sys
from pathlib import Path

pkg_path = str(Path(__file__).parent.parent)
sys.path.append(pkg_path)

import torch as th
from src.configs import InpaintingConfig, TPSConfig
from src.dense_motion import DenseMotionConf, DenseMotionNetwork, DenseMotionResult
from src.inpainting import InpaintingNetwork
from src.kp_detection import KPDetector, KPResult


def test_all():
    K = 10
    N = 5
    batch = 16

    dense_motion_cfg = DenseMotionConf(
        base_dim=64, tps=TPSConfig(10, 5), num_blocks=3, in_features=3, max_features=256
    )
    inpaint_cfg = InpaintingConfig(
        base_dim=dense_motion_cfg.base_dim,
        in_features=3,
        num_down_blocks=3,
        num_occlusion_masks=dense_motion_cfg.num_occlusion_masks,
        max_features=dense_motion_cfg.max_features,
    )

    kpd = KPDetector(dense_motion_cfg.tps)
    dense = DenseMotionNetwork(dense_motion_cfg)
    inpaint = InpaintingNetwork(inpaint_cfg)

    src_img = th.randn((batch, 3, 128, 128))
    drv_img = th.randn((batch, 3, 128, 128))
    src_res = kpd(src_img)
    assert tuple(src_res.foregroud_kp.shape) == (batch, K, N, 2)
    src_res: KPResult
    drv_res = kpd(drv_img)
    assert tuple(drv_res.foregroud_kp.shape) == (batch, K, N, 2)
    drv_res: KPResult

    res = dense(src_img, src_res, drv_res, dropout_prob=0.15)
    res: DenseMotionResult

    res = inpaint(src_img, res.occlusion_masks, res.optical_flow)
