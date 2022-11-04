from typing import Dict, Sequence

import imageio
import numpy as np
import torch as th
import torchvision
from einops import rearrange, repeat
from PIL import Image
from skimage.transform import resize

from transmotion.data_loading import map_numpy, video_iterator
from transmotion.dense_motion import BGMotionPredictor, DenseMotionNetwork
from transmotion.inpainting import InpaintingNetwork
from transmotion.kp_detection import KPDetector
from transmotion.viz import draw_points_on_tensors, show_on_gird

t_to_pil = torchvision.transforms.ToPILImage()
pil_to_t = torchvision.transforms.ToTensor()


def drv_vid_tensor(img_size: int):
    drv_vid = "../static/driving.mp4"
    all_frames = np.stack(
        list(
            map_numpy(
                lambda f: resize(f, (img_size, img_size)),
                lambda f: f.astype(np.float32),
                it=video_iterator(drv_vid),
            )
        )
    )
    drv_vid = rearrange(th.from_numpy(all_frames), "b h w d -> b d h w")

    return drv_vid


def src_img_tensor(img_size: int):
    src_img_path = "../static/source.png"

    src_img = resize(imageio.imread(src_img_path), (img_size, img_size)).astype(
        np.float32
    )

    return repeat(th.from_numpy(src_img), "h w d -> bs d h w", bs=1)


def pretrained_weights_to_model_cls(fpath: str) -> Dict[th.nn.Module, Dict]:
    _name_to_new_module = {
        "kp_detector": KPDetector,
        "dense_motion_network": DenseMotionNetwork,
        "inpainting_network": InpaintingNetwork,
        "bg_predictor": BGMotionPredictor,
    }

    saved_weights = th.load(fpath, map_location=th.device("cpu"))

    res = {}
    for k, mod in _name_to_new_module.items():
        res[mod] = saved_weights[k]

    return res


def tensors_to_pils(
    *ts: Sequence[th.Tensor], size: int = None
) -> Sequence[Image.Image]:
    pils = [t_to_pil(t.squeeze(0)) for t in ts]
    if size:
        pils = [p.resize((size, size)) for p in pils]
    return pils


def pils_to_grid(*pils: Sequence[Image.Image], size: int) -> Image.Image:
    tens = th.stack([pil_to_t(pim.resize((size, size))) for pim in pils], dim=0)
    return show_on_gird(tens)


def optical_flow_pil(flow: th.Tensor, size: int = None) -> Image.Image:
    flow_pil = t_to_pil(torchvision.utils.flow_to_image(flow.permute(0, 3, 1, 2)[0]))
    if size:
        flow_pil = flow_pil.resize((size, size))
    return flow_pil
