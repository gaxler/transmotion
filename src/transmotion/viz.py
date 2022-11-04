from typing import Sequence
import numpy as np
import torch as th
import torchvision

from PIL import Image, ImageDraw 
from PIL.ImageColor import colormap

from skimage import img_as_ubyte

from .kp_detection import KPResult


to_t = torchvision.transforms.ToTensor()

def draw_points_on_tensors(img_tensor: th.Tensor, key_points: KPResult, radius: int = 2) -> Sequence[Image.Image]:
    """
    Draw predicted key-points on images
    :param img_tensor: An image tenosr assumed to be in $[0,1]$ and shape of $(batch_size, dim, h, w)$
    :param key_points: Result on predictoin from 
    """


    # TODO: Kinda assume we only work with square images. (this is fine for now, the whole repo is built under this assumption)
    bs, d, h, w = img_tensor.shape

    color_list = list(colormap.keys())

    # shape: bs K N 2
    abs_keypoints = (
        key_points.detach_to_cpu().foreground_kp.add(1).div(2).mul(h)
    )
    abs_keypoints = abs_keypoints.numpy().astype(np.int16)


    return_pil_imgs = []

    for (img, batch_kp) in zip(img_tensor, abs_keypoints):
        new_img = img.clone().permute(1,2,0).detach().cpu().numpy()
        pil_im = Image.fromarray(img_as_ubyte(new_img))
        draw = ImageDraw.Draw(pil_im)
        # shape kp: N 2
        for tps_idx, kp in enumerate(batch_kp):
            color_idx = tps_idx % len(color_list)
            color = color_list[color_idx]
            for kpt in kp:
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=None, width=0)
        
        return_pil_imgs.append(pil_im)

    return return_pil_imgs


def show_on_gird(*img_tnesors, **kwargs):
    """
    kwargs are passed to torchvision.utils.make_grid
    """
    grid_tensor = th.cat(img_tnesors, dim=0)
    out_example = torchvision.utils.make_grid(grid_tensor, **kwargs).permute(1, 2, 0).numpy()
    return Image.fromarray(img_as_ubyte(out_example))

def show_points_on_grid(img_tensor: th.Tensor, key_points: KPResult, radius: int = 2) -> Image.Image:
    pil_imgs = draw_points_on_tensors(img_tensor=img_tensor, key_points=key_points, radius=radius)
    kp_drawn_tensors = [to_t(pim).unsqueeze(0) for pim in pil_imgs]
    return show_on_gird(*kp_drawn_tensors)

