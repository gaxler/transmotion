from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class TPSConfig:
    num_tps: int
    points_per_tps: int


@dataclass
class DenseMotionConf:
    """
    :param base_dim: Dimension that encoder and decoder networks start from. This dim multiplied in each block i by a factor of 2^i
    :param in_features: Input dimension to the dense mottion network. Usually the input is an image so this value is 3
    """

    base_dim: int
    tps: TPSConfig
    num_blocks: int
    in_features: int
    max_features: int
    kp_variance: float = 0.01
    scale_factor: float = 0.25
    num_occlusion_masks: int = 4


@dataclass
class InpaintingConfig:
    base_dim: int
    in_features: int
    num_down_blocks: int
    num_occlusion_masks: int
    max_features: int

    def __post_init__(self):
        self._occlusion_masks_consistent()

    def _occlusion_masks_consistent(self):
        _msg = f"Occlusion masks must be one more than down blocks (or a single mask) got {self.num_occlusion_masks} massk and {self.num_down_blocks} blocks"
        assert (
            self.num_occlusion_masks == self.num_down_blocks + 1
        ) or self.num_occlusion_masks == 1, _msg


@dataclass
class PerceptualLoss:
    """
    Get VGG features from multiple scales of the driving image and the generated image.
    Train on generated images with L_1 loss.

    :param scales: List of float number that represent the scaling factors to be applied on the images. This needs to conform to the number of feature maps you get from the perceptual loss network.
    By default we use VGG19 here, this net has 5 feature map output. #TODO: Might need to make this more general and explicitly link number of feature maps to number of scales.
    :param loss_weight: Loss weights (for each scale) when aggreagted into total loss value
    :param in_dim: Perceptual loss multi-scale image pyramid, this is the in dimension of the pyramid input
    """

    scales: Sequence[float]
    loss_weights: Sequence[float]
    in_dim: int


@dataclass
class EquivarianceLoss:
    sigma_tps: float
    sigma_affine: float
    points_per_tps: int
    loss_weight: float


@dataclass
class WarpLoss:
    loss_weight: float


@dataclass
class BackgroundLoss:
    loss_weight: float


@dataclass
class LossConfig:
    perceptual: PerceptualLoss
    equivariance: EquivarianceLoss
    warp: WarpLoss
    background: BackgroundLoss


@dataclass
class OptimizerConfig:
    """
    :param initial_lr: base learning rate for the optimizer
    :param lr_decay_epoch_sched: Sequence of epoch numbers that represent learning rate decay steps. decay is by lr_decay_gamma
    :param lr_decay_gamma: Learning rate decay parameter. Decay happens according to lr_decay_epoch_sched
    """

    initial_lr: float
    lr_decay_epoch_sched: Sequence[int]
    """ Sequence of epoch numbers that represent learning rate decay steps. decay is by lr_decay_gamma """

    lr_decay_gamma: float
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4


@dataclass
class DropoutConfig:
    """
    Drop-out is applied during tarining to the DenseMotion network. Its not a property of the network but an external param to the forward pass.
    So the config for that is in general train config and not the dense motion config.

    :param prob_inc_epoch: Increment dropout probaility over this amount of epochs
    """

    start_epoch: int
    init_prob: float
    max_prob: float
    prob_inc_epochs: int = 10


@dataclass
class DataLoadingConfig:
    batch_size: int
    num_workers: int


@dataclass
class TrainConfig:
    """ """

    data_loading: DataLoadingConfig
    num_epochs: int
    optimizer: OptimizerConfig
    dropout: DropoutConfig
    dense_motion: DenseMotionConf
    inpainting: InpaintingConfig
    tps: TPSConfig
    loss: LossConfig
    background_start_epoch: int
    image_size: int


def dummy_conf() -> TrainConfig:
    """VOX model dummy config"""
    dl_conf = DataLoadingConfig(batch_size=16, num_workers=6)

    opt_conf = OptimizerConfig(
        initial_lr=1e-4, lr_decay_epoch_sched=[40, 70], lr_decay_gamma=0.1
    )
    do_conf = DropoutConfig(
        start_epoch=30, init_prob=0.0, max_prob=0.3, prob_inc_epochs=10
    )
    tps_conf = TPSConfig(num_tps=10, points_per_tps=5)
    dense_conf = DenseMotionConf(
        base_dim=64, tps=tps_conf, num_blocks=5, in_features=3, max_features=1024
    )

    inpaint_conf = InpaintingConfig(
        base_dim=dense_conf.base_dim,
        in_features=dense_conf.in_features,
        num_down_blocks=3,
        num_occlusion_masks=dense_conf.num_occlusion_masks,
        max_features=512,
    )

    loss_conf = LossConfig(
        perceptual=PerceptualLoss(
            scales=[0.5, 0.25, 0.125], loss_weights=[10, 10, 10, 10, 10], in_dim=3
        ),
        equivariance=EquivarianceLoss(
            sigma_tps=0.005,
            sigma_affine=0.05,
            points_per_tps=tps_conf.points_per_tps,
            loss_weight=10,
        ),
        warp=WarpLoss(loss_weight=10),
        background=BackgroundLoss(loss_weight=10),
    )

    return TrainConfig(
        data_loading=dl_conf,
        num_epochs=10,
        optimizer=opt_conf,
        dropout=do_conf,
        dense_motion=dense_conf,
        inpainting=inpaint_conf,
        tps=tps_conf,
        loss=loss_conf,
        background_start_epoch=30,
        image_size=256,
    )
