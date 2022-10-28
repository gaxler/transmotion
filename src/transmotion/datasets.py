"""
In training you sample a video, and sample two frames out of the video, one of the frames serves as the source and the other as driving.

we get a video array, in training this will be two frames out of the video, in reconstruction mode, we take all of the frames.

- they load the whole video to get only two frames out of it. in demo run, the use all of the frames.
pretty simple actually
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import torch as th

from torch.utils.data._utils.collate import default_collate


@dataclass
class SourceDrivingSample:
    """
    :param source: source image, shape [d (=3), h, w]
    :param driving: source image, shape [d (=3), h, w]
    """

    source: th.Tensor
    driving: th.Tensor


@dataclass
class SourceDrivingBatch:
    """
    :param source: source image, shape [bs, d (=3), h, w]
    :param driving: source image, shape [bs, d (=3), h, w]
    """

    source: th.Tensor
    driving: th.Tensor


def collate_fn(batch: Sequence[SourceDrivingSample]) -> SourceDrivingBatch:
    srcs, drvs = [], []
    for samp in batch:
        srcs.append(samp.source)
        drvs.append(samp.driving)

    return SourceDrivingBatch(
        source=default_collate(srcs), driving=default_collate(drvs)
    )
