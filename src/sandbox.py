from typing import Iterable, List, Sequence, Tuple

import imageio
import torch as th
from einops import rearrange, repeat
from skimage import img_as_ubyte
from skimage.transform import resize
from torch.utils.data import DataLoader

from transmotion.blocks import ImagePyramid
from transmotion.configs import (
    BackgroundLoss,
    DataLoadingConfig,
    DropoutConfig,
    EquivarianceLoss,
    OptimizerConfig,
    PerceptualLoss,
    TPSConfig,
    TrainConfig,
    WarpLoss,
)
from transmotion.datasets import SourceDrivingBatch
from transmotion.dense_motion import (
    BGMotionParam,
    BGMotionPredictor,
    DenseMotionConf,
    DenseMotionNetwork,
    DenseMotionResult,
)
from transmotion.helpers import (
    NetworksBundle,
    chain_param,
    get_kp_normalized_forward_func,
    load_original_weights,
)
from transmotion.inpainting import InpaintingConfig, InpaintingNetwork, InpaintingResult
from transmotion.kp_detection import KPDetector, KPResult
from transmotion.nn_blocks import get_pretrained_vgg19
from transmotion.utils import ThinPlateSpline, deform_with_4d_deformation, grid_like_featuremap

Optimizers = Sequence[th.optim.Optimizer | None]
Schedulers = Sequence[th.optim.lr_scheduler.MultiStepLR | None]


def build_optimizers_and_schedulers(
    opt_cfg: OptimizerConfig, nets: NetworksBundle, last_epoch: int = -1
) -> Tuple[Optimizers, Schedulers]:
    """
    Seems that the only reason original work has two optimizers is because bg_predictor not always present
    """

    main_params = {
        "params": chain_param(nets.key_points, nets.dense_motion, nets.inpaint),
        "initial_lr": opt_cfg.initial_lr,
    }
    bg_params = (
        {
            "params": nets.background_motion.parameters(),
            "initial_lr": opt_cfg.initial_lr,
        }
        if nets.background_motion is not None
        else None
    )

    def _make_opt(group):
        if group is None:
            return None
        return th.optim.Adam(
            params=[group],
            lr=opt_cfg.initial_lr,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            weight_decay=opt_cfg.weight_decay,
        )

    def _make_sched(opt):
        if opt is None:
            return None
        return th.optim.lr_scheduler.MultiStepLR(
            optimizer=opt,
            milestones=opt_cfg.lr_decay_epoch_sched,
            gamma=opt_cfg.lr_decay_gamma,
            last_epoch=last_epoch,
        )

    opts = (_make_opt(main_params), _make_opt(bg_params))
    scheds = tuple(_make_sched(opt) for opt in opts)
    return opts, scheds


class LossFuncBundle:
    """
    Perceptual loss needs a VGG19 and some anit-alias downsampling. Both those objects are held by this class.
    The rest of the losses are bundeled in here to
    """

    def __init__(
        self,
        perceptual: PerceptualLoss,
        equivariance: EquivarianceLoss,
        warp_loss: WarpLoss,
        background_loss: BackgroundLoss,
    ) -> None:

        self.img_pyramide = ImagePyramid(
            perceptual.scales, in_dim=perceptual.in_dim
        )  # return
        (
            self.vgg,
            self.num_vgg_fmaps,
        ) = get_pretrained_vgg19()  # this returns List[th.Tensor] of len == 5

        self.perceptual_cfg = perceptual
        self.equivar_cfg = equivariance
        self.warp_cfg = warp_loss
        self.bkd_cfg = background_loss

    def perceptual_losses(
        self, driving: th.Tensor, generated: th.Tensor
    ) -> Tuple[Sequence[th.Tensor], Sequence[float]]:

        driving_pyramid_seq = self.img_pyramide(driving)
        gen_pyramid_seq = self.img_pyramide(generated)
        perceptual_losses = [0 for _ in range(self.num_vgg_fmaps)]
        for drv, gen in zip(driving_pyramid_seq, gen_pyramid_seq):
            drv_vgg: List[th.Tensor]
            drv_vgg = self.vgg(drv)

            gen_vgg: List[th.Tensor]
            gen_vgg = self.vgg(gen)

            # TODO: loss weights depend on the output of the external VGG19 model!
            for fmap_idx, (dv, gv) in enumerate(zip(drv_vgg, gen_vgg)):
                current_scale_loss = th.abs(dv.detach() - gv).mean()
                perceptual_losses[fmap_idx] += current_scale_loss

        return perceptual_losses, self.perceptual_cfg.loss_weights

    def equivariance_loss(
        self, driving_img: th.Tensor, drv_kp: KPResult, kp_detector: KPDetector
    ) -> Tuple[th.Tensor, float]:

        cfg = self.equivar_cfg
        bs, d, h, w = driving_img.shape

        random_tps = ThinPlateSpline.random(
            num_points=cfg.points_per_tps,
            sigma_affine=cfg.sigma_affine,
            sigma_tps=cfg.sigma_tps,
            batch_size=bs,
        )

        coords = grid_like_featuremap(driving_img)
        # this returns coordinates with 1 TPS. we want to get rid of the TPS,
        # since we don't really need it here
        deform_cords = random_tps.warp_cords(coordinates=coords).squeeze(1)

        deformed_driving_img = deform_with_4d_deformation(
            driving_img, deformation=deform_cords
        )

        deformed_kp = kp_detector(deformed_driving_img)
        deformed_kp: KPResult

        warpped_deformed_kp = random_tps.warp_cords(
            deformed_kp.foreground_kp
        ).squeeze()  # [bs, K, N, 2] -> [bs, 1, K, N, 2] -> [bs, K, N, 2]

        loss = th.abs(drv_kp.foreground_kp - warpped_deformed_kp).mean()
        return loss, self.equivar_cfg.loss_weight

    def warp_occlude_optical_flow_loss(
        self,
        driving_img: th.Tensor,
        occlusion_masks: Sequence[th.Tensor],
        deformed_src_fmaps: Sequence[th.Tensor],
        inpaint: InpaintingNetwork,
    ) -> Tuple[th.Tensor, float]:

        drv_warp_encodings = inpaint.get_encode_no_grad(
            driver_img=driving_img, occlusion_masks=occlusion_masks
        )
        loss = 0
        for src, drv in zip(deformed_src_fmaps, drv_warp_encodings):
            loss += th.abs(src - drv).mean()

        return loss, self.warp_cfg.loss_weight

    def background_transform_consistency_loss(
        self,
        src_img: th.Tensor,
        drv_img: th.Tensor,
        bg_param: BGMotionParam,
        bkg_motion_pred: BGMotionPredictor,
    ) -> Tuple[th.Tensor, float]:
        rev_bg_param = bkg_motion_pred(drv_img, src_img)
        rev_bg_param: BGMotionParam

        concat_order_is_inverse_transpose = th.einsum(
            "bvd,bdv->bvv", bg_param.bg_params, rev_bg_param.bg_params
        )
        ident = th.eye(3, dtype=bg_param.bg_params.dtype).to(src_img.device)
        loss = th.abs(ident - concat_order_is_inverse_transpose).mean()
        return loss, self.bkd_cfg.loss_weight


def step(
    src_img: th.Tensor,
    drv_img: th.Tensor,
    nets: NetworksBundle,
    losses: LossFuncBundle,
    dropout_prob: float,
) -> th.Tensor:
    kp_src = nets.key_points(src_img)
    kp_drv = nets.key_points(drv_img)

    bg_param: BGMotionParam
    bg_param = (
        nets.background_motion(src_img, drv_img)
        if nets.background_motion is not None
        else None
    )

    motion_res: DenseMotionResult
    motion_res = nets.dense_motion(
        source_img=src_img,
        src_kp=kp_src,
        drv_kp=kp_drv,
        bg_param=bg_param,
        dropout_prob=dropout_prob,
    )

    inpaint_res: InpaintingResult
    inpaint_res = nets.inpaint(
        source_img=src_img,
        occlusion_masks=motion_res.occlusion_masks,
        optical_flow=motion_res.optical_flow,
    )

    from torchvision.utils import save_image

    print("saving img")
    save_image(inpaint_res.inpainted_img.detach().cpu(), "batch.jpg")

    weighted_loss = 0
    percep_loss, percep_weight = losses.perceptual_losses(
        driving=drv_img, generated=inpaint_res.inpainted_img
    )

    for l, w in zip(percep_loss, percep_weight):
        weighted_loss += l * w

    equi_loss, equi_weight = losses.equivariance_loss(
        driving_img=drv_img, drv_kp=kp_drv, kp_detector=nets.key_points
    )

    weighted_loss += equi_loss * equi_weight

    warp_loss, warp_weight = losses.warp_occlude_optical_flow_loss(
        driving_img=drv_img,
        occlusion_masks=motion_res.occlusion_masks,
        deformed_src_fmaps=inpaint_res.deformed_src_fmaps,
        inpaint=nets.inpaint,
    )

    weighted_loss += warp_loss * warp_weight

    if bg_param is not None:
        bg_loss, bg_weight = losses.background_transform_consistency_loss(
            src_img=src_img,
            drv_img=drv_img,
            bg_param=bg_param,
            bkg_motion_pred=nets.background_motion,
        )

        weighted_loss += bg_loss * bg_weight

    return weighted_loss


def _get_dropout_prob(epoch: int, params: DropoutConfig) -> float:
    if epoch < params.start_epoch:
        return 0.0

    diff = params.max_prob - params.init_prob
    inc_end_epoch = params.start_epoch + params.prob_inc_epochs
    ratio = min(inc_end_epoch, epoch - params.start_epoch) / params.prob_inc_epochs
    prob = params.init_prob + diff * ratio

    return prob


def epoch(
    epoch_num: int,
    data_loader: Iterable[SourceDrivingBatch],
    nets: NetworksBundle,
    losses: LossFuncBundle,
    optimizers: Sequence[th.optim.Adam | None],
    dropout_prob: float = 0.0,
    max_grad_norm: float = 10.0,
):
    def _clip_grads_for_net(net: th.nn.Module | None):
        if net is not None:
            th.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=max_grad_norm, norm_type="inf"
            )

    for batch in data_loader:
        loss = step(
            src_img=batch.source,
            drv_img=batch.driving,
            nets=nets,
            losses=losses,
            dropout_prob=dropout_prob,
        )
        print(f"{loss.item()}")
        loss.backward()

        _clip_grads_for_net(nets.key_points)
        _clip_grads_for_net(nets.dense_motion)
        _clip_grads_for_net(nets.background_motion)

        for opt in optimizers:
            if opt is None:
                continue
            opt.step()

        # Clear grad by using pytorch memory allocator
        nets.clear_grads()

        pass


def get_dataloader(conf: DataLoadingConfig) -> DataLoader:
    import numpy as np

    from transmotion.data_loading import _center_crop_to_size, map_numpy, video_iteraotr

    drv_vid = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/driving.mp4"
    # src_img = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/source.png"
    all_frames = np.stack(
        map_numpy(
            _center_crop_to_size(384),
            lambda f: f.astype(np.float32),
            it=video_iteraotr(drv_vid),
        )
    )
    all_frames = np.divide(all_frames, 255.0)
    N = all_frames.shape[0]
    for _ in range(1):
        idxs = np.random.choice(N, 2 * conf.batch_size)
        src_idx, drv_idx = idxs[: conf.batch_size], idxs[conf.batch_size :]
        src_img = rearrange(th.from_numpy(all_frames[src_idx]), "b h w d -> b d h w")
        drv_img = rearrange(th.from_numpy(all_frames[drv_idx]), "b h w d -> b d h w")
        yield SourceDrivingBatch(source=src_img, driving=drv_img)


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


def train(conf: TrainConfig, last_epoch: int = -1, checkpoint_path: str | None = None):
    dataloader = get_dataloader(conf=conf.data_loading)
    nets = build_nets(tps=conf.tps, dense=conf.dense_motion, inpaint=conf.inpainting)
    if checkpoint_path:
        nets = load_original_weights(checkpoint_path, nets)

    losses = LossFuncBundle(
        perceptual=conf.loss.perceptual,
        equivariance=conf.loss.equivariance,
        warp_loss=conf.loss.warp,
        background_loss=conf.loss.background,
    )

    (main_opt, bg_opt), (main_sched, bg_sched) = build_optimizers_and_schedulers(
        opt_cfg=conf.optimizer, nets=nets, last_epoch=last_epoch
    )

    for epoch_idx in range(conf.num_epochs):
        nets.clear_grads()

        train_bg = conf.background_start_epoch < epoch_idx
        do_prob = _get_dropout_prob(epoch=epoch_idx, params=conf.dropout)

        optimizers = (main_opt, bg_opt) if train_bg else (main_opt, None)

        epoch(
            epoch_num=epoch_idx,
            data_loader=dataloader,
            nets=nets if train_bg else nets.no_background(),
            losses=losses,
            optimizers=optimizers,
            dropout_prob=do_prob,
        )

        main_sched.step()
        if train_bg:
            bg_sched.step()
    return


if __name__ == "__main__":
    import numpy as np
    import torchvision
    from PIL import Image

    from transmotion.configs import dummy_conf
    from transmotion.data_loading import map_numpy, video_iteraotr, show_on_gird

    conf = dummy_conf()

    SIZE = conf.image_size

    def _drv_vid_tensor():
        drv_vid = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/driving.mp4"
        all_frames = np.stack(
            map_numpy(
                # lambda f: np.asarray(
                #     Image.fromarray(f).resize(
                #         (SIZE, SIZE), resample=Image.Resampling.LANCZOS
                #     )
                # ),
                lambda f: resize(f, (SIZE, SIZE)),
                lambda f: f.astype(np.float32),
                it=video_iteraotr(drv_vid),
            )
        )
        # all_frames = np.divide(all_frames, 255.0)
        drv_vid = rearrange(th.from_numpy(all_frames), "b h w d -> b d h w")

        return drv_vid

    def _src_img_tensor():
        src_img_path = "/Users/gregoryaxler/Desktop/projects/Thin-Plate-Spline-Motion-Model/assets/source.png"

        # src_pil = (
        #     Image.open(src_img_path)
        #     .convert("RGB")
        #     .resize((SIZE, SIZE), resample=Image.Resampling.LANCZOS)
        # )
        # src_img = np.asarray(src_pil).astype(np.float32)
        # src_img = np.divide(src_img, 255.0)

        src_img = resize(imageio.imread(src_img_path), (SIZE, SIZE)).astype(np.float32)

        return repeat(th.from_numpy(src_img), "h w d -> bs d h w", bs=1)

    nets = build_nets(tps=conf.tps, dense=conf.dense_motion, inpaint=conf.inpainting)
    nets = load_original_weights("vox.pth.tar", nets)
    nets = nets.eval()

    src_img = _src_img_tensor()

    initial_driving_kp = None
    predictions = []
    drv_frame_infer_fn = None
    for idx, drv_img in enumerate(_drv_vid_tensor()[:50]):
        if drv_frame_infer_fn is None:
            drv_frame_infer_fn = get_kp_normalized_forward_func(
                src_img=src_img,
                intial_drv_img=drv_img.unsqueeze(0),
                nets=nets,
            )
        res = drv_frame_infer_fn(drv_img.unsqueeze(0))
        frame = rearrange(res.generated_image, "b d h w -> b h w d").squeeze(0)
        predictions.append(frame.numpy())

    imageio.mimsave("out.mp4", [img_as_ubyte(f) for f in predictions], fps=25)
