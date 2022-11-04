import os
import sys
from pathlib import Path

pkg_path = Path(__file__).parent.parent
print(str(pkg_path))
sys.path.append(str(pkg_path))

import pytest
import imageio
import numpy as np
import torch as th
from tqdm import tqdm
from einops import rearrange, repeat
from skimage.transform import resize
from skimage import img_as_ubyte
from transmotion.configs import dummy_conf 
from transmotion.data_loading import map_numpy, video_iterator
from transmotion.network_bundle import (
    NetworksBundle,
    build_nets,
    get_kp_normalized_forward_func,
)

from transmotion.kp_detection import KPDetector, KPResult

STATIC_PATH = pkg_path.parent / "static"
WEIGHTS_PATH = STATIC_PATH / "vox.pth.tar"


def _drv_vid_tensor(img_size: int):
    drv_vid = str(STATIC_PATH / "driving.mp4")

    frame_pipeline = map_numpy(
        lambda f: resize(f, (img_size, img_size)),
        lambda f: f.astype(np.float32),
        it=video_iterator(drv_vid),
    )
    all_frames = np.stack(list(frame_pipeline))

    drv_vid = rearrange(th.from_numpy(all_frames), "b h w d -> b d h w")

    return drv_vid


def _src_img_tensor(img_size: int):
    src_img_path = str(STATIC_PATH / "source.png")

    src_img = resize(imageio.imread(src_img_path), (img_size, img_size)).astype(
        np.float32
    )

    return repeat(th.from_numpy(src_img), "h w d -> bs d h w", bs=1)


class NetworkState:
    nets: NetworksBundle

    def __init__(self) -> None:
        self.conf = dummy_conf()
        self.src_img = _src_img_tensor(img_size=self.conf.image_size)
        self.drv_vid = _drv_vid_tensor(img_size=self.conf.image_size)

    def build_nets(self, fpath: str):
        self.nets = build_nets(
            tps=self.conf.tps,
            dense=self.conf.dense_motion,
            inpaint=self.conf.inpainting,
        )
        self.nets.load_original_weights(fpath=str(WEIGHTS_PATH))


state = NetworkState()


@pytest.mark.dependency()
def test_network_build():
    assert WEIGHTS_PATH.exists(), f"No vox.pth.tar in {str(STATIC_PATH)}"
    state.build_nets(str(WEIGHTS_PATH))


@pytest.mark.dependency(depends=["test_network_build"])
def test_vid_inference():
    nets = state.nets.eval()

    predictions = []
    drv_frame_infer_fn = None
    for idx, drv_img in tqdm(enumerate(state.drv_vid)):
        if drv_frame_infer_fn is None:
            drv_frame_infer_fn = get_kp_normalized_forward_func(
                src_img=state.src_img,
                intial_drv_img=drv_img.unsqueeze(0),
                nets=nets,
            )
        res = drv_frame_infer_fn(drv_img.unsqueeze(0))

        gen_frame = rearrange(res.generated_image, "b d h w -> b h w d").squeeze(0)
        drv_frame = rearrange(drv_img, "d h w -> h w d")
        frame = th.cat([gen_frame, drv_frame], dim=1)
        predictions.append(frame.detach().cpu().numpy())

    out_path = str(STATIC_PATH / "out.gif")
    imageio.mimsave(out_path, [img_as_ubyte(f) for f in predictions], fps=25)