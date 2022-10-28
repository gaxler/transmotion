from typing import Callable, Iterable

import imageio
import imageio.v3 as iio
from skimage.transform import resize
from skimage import img_as_ubyte
import numpy as np
from einops import rearrange, repeat
import torch as th
import torchvision
from PIL import Image

from .kp_detection import KPResult


def video_iteraotr(fpath: str) -> Iterable[np.ndarray]:
    """
    Turn a video file into an iterator over frames
    .. note:: This uses imageio for video decoding.

    :param fpath: Path to video file
    """

    # reader = iio.imiter(fpath, plugin="pyav")
    reader = imageio.get_reader(fpath)
    for frame in reader:
        yield frame


def load_driving_video(vid_path: str):
    import mediapipe

    face_detector = mediapipe.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    meta = []
    frames = []
    for frame in video_iteraotr(vid_path):
        frames.append(frame)
        meta.append(face_detector.process(frame))

    return frames, meta


def _center_crop_to_size(size: int) -> np.ndarray:
    def crop_fn(frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        dh = max(0, h - size) // 2
        dw = max(0, w - size) // 2

        return frame[dh:-dh, dw:-dw, :]

    return crop_fn


def map_numpy(
    *funcs: Callable[[np.ndarray], np.ndarray], it: Iterable[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Map a sequence of function that take numpy array and return a numpy array over an iterator of numpy arrays.
    
    This is a helper function to apply transformations on the video iterator.
    """
    for fn in funcs:
        it = map(fn, it)
    return it


def _to_tensor(it: Iterable[np.ndarray], frame_size: int) -> th.Tensor:

    all_frames = np.stack(
        list(
            map_numpy(
                lambda f: resize(f, (frame_size, frame_size)),
                lambda f: f.astype(np.float32),
                it=it,
            )
        )
    )
    _tensor = rearrange(th.from_numpy(all_frames), "b h w d -> b d h w")

    return _tensor


def video_to_tensor(vid_path: str, frame_size: int):
    return _to_tensor(it=video_iteraotr(vid_path), frame_size=frame_size)


def image_to_tensor(img_path: str, frame_size: int) -> th.Tensor:

    src_img = resize(iio.imread(img_path), (frame_size, frame_size)).astype(np.float32)

    return repeat(th.from_numpy(src_img), "h w d -> bs d h w", bs=1)


if __name__ == "__main__":
    # TODO: replace this with torchvision video loading, if gets out of beta
    drv_vid = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/driving.mp4"
    src_img = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/source.png"
    cropped_vid_it = map_numpy(_center_crop_to_size(384), it=video_iteraotr(drv_vid))
    for frame in cropped_vid_it:
        pass
