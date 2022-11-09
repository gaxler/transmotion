"""
Thin-Plate Spline Transform implementation and some helper functions to work with points and grids.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat


_eps = 1e-2
_nano_eps = 1e-9


def conform_to_type_and_device(leader: th.Tensor) -> Callable[[th.Tensor], th.Tensor]:
    def _cast_to_device(t: th.Tensor):
        """cast dtype of Tensor to that of the leader Tensor and move tensor to device of leader"""
        return t.type(leader.type()).to(leader.device)

    return _cast_to_device


@th.jit.script
def fused_dist_square(fst: th.Tensor, snd: th.Tensor) -> th.Tensor:
    """Get l2 norm squared over the last dim (JITTed function to fuse simple operations)"""
    return fst.sub(snd).pow(2).sum(-1)


@th.jit.script
def fused_basis_func(vals: th.Tensor) -> th.Tensor:
    # vals has distances squared in it
    # return vals.mul(th.log(vals.sqrt() + 1e-9))

    # TODO: looks like the code has a mistake in how in calculates the basis function. i need to repeat the mistake for reproducability
    # Probably nothing, it adds a factor of two to the distance func. nets probably learn to adjust to it (maybe it was intentional?)
    _nano_eps = 1e-9
    return vals.mul(th.log(vals.add(_nano_eps)))


def make_coordinate_grid(
    height: int, width: int, dtype: th.dtype = th.float32
) -> th.Tensor:
    """
    Create a meshgrid `[-1,1] x [-1,1]` with shape of `[height, width, 2]`

    :return: shape [height, width, 2]
    """

    x = th.linspace(-1, 1, steps=width, dtype=dtype)
    y = th.linspace(-1, 1, steps=height, dtype=dtype)
    meshed = th.stack(
        (repeat(x, "w -> h w", h=height), repeat(y, "h -> h w", w=width)), dim=2
    )
    return meshed


def grid_like_featuremap(fmap: th.Tensor) -> th.Tensor:
    """
    Generate coordinate grid with same spatial size, type and device as fmap
    :param fmap: Feature map tensor to be used as spatial shape.
    .. note::
       Assumes last two dimension are height and width
    :type fmap: [bs, d. h, w]
    """

    bs, d, h, w = fmap.shape

    grid = make_coordinate_grid(h, w, dtype=fmap.dtype).unsqueeze(0)
    grid = grid.to(fmap.device)
    return grid


def deform_with_5d_deformation(frame: th.Tensor, deformation: th.Tensor) -> th.Tensor:
    """
    Deform image frame according to wrapped coordinates.
    The deformation is done with PyTorch's ``grid_sample`` that takes 4D tensors as deformation coordinates

    :param deformation: Batch of K coordinates grids that describe where each pixel should go
    :type frame: [bs, d, h, w]
    :type deformation: [bs, K, h, w, d]
    """
    b, kn_o, h, w, d = deformation.shape
    frame = repeat(frame, "b d h w -> b kn_o d h w", kn_o=kn_o)
    input_ = rearrange(frame, "b k d h w -> (b k) d h w")
    cords = rearrange(deformation, "b kn_o h w d -> (b kn_o) h w d")
    deformed_frame = F.grid_sample(input_, cords, align_corners=True)
    deformed_frame = rearrange(
        deformed_frame, "(b kn_o) d h w -> b kn_o d h w", kn_o=kn_o
    )
    return deformed_frame


def deform_with_4d_deformation(frame: th.Tensor, deformation: th.Tensor) -> th.Tensor:
    """
    Apply pythorch's ``grid_sample`` with ``align_corners``.

    :param frame: shape [b, d, h, w]
    :param deformation: shape [b, h, w, 2]
    """
    deformed_frame = F.grid_sample(frame, deformation, align_corners=True)
    return deformed_frame


@dataclass
class ThinPlateSpline:
    """
    This Spline is equivalent ot a function that transforms a set of point to be close to some other set of point while keeping the transformation smooth.
    The other set of points are the control points and smoothness is measured as the second derivative of the transformation.

    we have the following params in this transformation:
    
    - :math:`A_k, b_k` = theta
    - :math:`W_k` = control params
    - :math:`p_k` = control points

    To transform a point P:

    .. math::
        \hat{P} = (A_k p + b_k) + W_k U(|| p_k - p ||)
    

    :math:`U` is the Kernel function. 


    """

    control_points: th.Tensor
    """ The points that our transformation needs to match exactly. Shape - ``[b, K, 2]`` Each new point we transform is am affine combination of control set point. each control point is weighted according to a kernel distance from the point being transformed """
    """ [bs, K, N, 2] control points for each TPS transform"""
    control_params: th.Tensor
    """ Weights that we apply on our kernel function """
    """ [bs, 1, K**2]"""
    """ Those are the [b K N 2]"""

    theta: th.Tensor
    """ Affine Transform Parameters [bs, K, 2, 3] a batch of K TPS transforms"""

    _warp_cords: Callable[[th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor]
    """ A function to warp coordinates, random warp uses a different wrapping implementation"""

    def warp_cords(self, coordinates: th.Tensor) -> th.Tensor:
        """
        :param: coordinates - `[bs, h, w, d]` coordinates to be wrapped
        :return: `[bs, K, h, w, d]`
        """
        _cast_to_device = conform_to_type_and_device(coordinates)

        theta = _cast_to_device(self.theta)
        control_params = _cast_to_device(self.control_params)
        control_points = _cast_to_device(self.control_points)

        return self._warp_cords(
            coordinates=coordinates,
            theta=theta,
            control_points=control_points,
            control_params=control_params,
        )

    @classmethod
    def random(
        cls,
        sigma_tps: float,
        num_points: int,
        sigma_affine: float,
        batch_size: int = 1,
        num_transforms: int = 1,
    ):
        """Random Transform is a special case of the full TPS. This one does a random affine transform and jitter them in a diagonal (weighted by TPS params)"""
        K = num_transforms
        noise = th.normal(mean=0, std=sigma_affine * th.ones([batch_size, K, 2, 3]))
        theta = noise + th.eye(2, 3)[None, None, :, :]  # [bs, K, 2, 3]

        control_points = make_coordinate_grid(
            num_points, num_points, dtype=noise.dtype
        )[
            None, ...
        ]  # [1, √N, √N, 2] this is just a convenience to spread points across the frame, we don't want it in a grid format.
        # this is not a random generation of points, so it stays the same for every batch member and every transform
        control_points = rearrange(
            control_points,
            "(b K) N_sqrt1 N_sqrt2 d -> b K (N_sqrt1 N_sqrt2) d",
            b=1,
            K=1,
        )  # [1, 1, N, 2]

        control_params = th.normal(
            mean=0,
            std=sigma_tps * th.ones([batch_size, K, num_points**2]),
        )  # [bs, K, N]

        def random_wrap_func(
            coordinates: th.Tensor,
            theta: th.Tensor,
            control_points: th.Tensor,
            control_params: th.Tensor,
        ) -> th.Tensor:
            """
            Warp coordinates according to a random TPS.
            :param: coordinates - `[bs, h, w, d]` coordinates to be wrapped
            :param: theta
            :param: control_points
            :param: control_params
            :return: return wrapped coordinates with an additional num_tps transforms axis. that is `[bs K h w d]`
            """

            # ed = exapnd dimension
            _cords = coordinates[:, :, :, None, None, :]  # [b h w 1 1 d]
            _ctrl_pts = control_points[:, None, None, :, :, :]  # [b 1 1 K N d]
            norms_sqr = fused_dist_square(_cords, _ctrl_pts)  # [bs h w K N]
            basis_func_values = fused_basis_func(norms_sqr)

            # [bs h w K N]*[bs 1 1 K N]
            # control_params = rearrange(control_params, "(b ed) K N -> b ed ed K N")
            control_params = control_params[:, None, None, :, :]  # [bs 1 1 K N]
            pre_jitter = basis_func_values * control_params  # [bs h w K N]
            # [bs K h w 1]
            jitter = rearrange(
                pre_jitter.sum(-1, keepdim=True), "b h w K N1 -> b K h w N1"
            )
            # jitter = rearrange(pre_jitter.sum(-1), "(b ed) h w K -> b K h w ed", ed=1)

            # affine transform
            _A = theta[:, :, :, :2]  # [b, K, 2, 2]
            _b = theta[:, :, :, 2:]  # [b, K, 2, 1]
            _b = rearrange(_b, "bs K v (h w) -> bs K h w v", h=1, w=1)

            transformed_points = (
                th.einsum("bkvd,bhwd->bkhwv", _A, coordinates) + _b
            )  # [bs K h w 2]

            wrapped_points = transformed_points + jitter

            return wrapped_points

        inst = cls(
            control_points=control_points,
            control_params=control_params,
            theta=theta,
            _warp_cords=random_wrap_func,
        )

        inst._warp_cords.__doc__ = random_wrap_func.__doc__
        return inst

    @classmethod
    def fit(cls, source_pts: th.Tensor, destination_pts: th.Tensor):
        """
        :param: source_pts - `[bs, num_tps, pts_per_tps, 2]` source points for TPS transform
        :param: destination_pts - `[bs, num_tps, pts_per_tps, 2]` destination points for TPS transform
        """
        bs, K, N, d = source_pts.shape
        assert (
            destination_pts.shape == source_pts.shape
        ), f"Source and destination shapes must be the same, got {source_pts.shape} & {destination_pts.shape}"
        dtype = source_pts.dtype
        device = source_pts.device
        """
        Q = 
        [  K  |  P ]
        [ P.T |  0 ]
        """
        Q_mat = th.zeros((bs, K, N + d + 1, N + d + 1), dtype=dtype).to(
            source_pts.device
        )

        # Calculate and populate K
        _K = th.norm(
            source_pts[:, :, :, None, :] - source_pts[:, :, None, :, :], dim=4, p=2
        )
        K_mat = _K.pow(2) * th.log(_K.pow(2) + _nano_eps)
        # K_mat = _K.pow(2) * th.log(_K + _nano_eps)

        Q_mat[:, :, :N, :N] = K_mat

        # Calculate P
        P_mat = th.ones((bs, K, N, d + 1), dtype=dtype, device=device)
        P_mat[:, :, :, :d] = source_pts

        # Populate P and P.T
        Q_mat[:, :, :N, N:] = P_mat  # P is [_ _ N 3]
        Q_mat[:, :, N:, :N] = rearrange(P_mat, "b k n d -> b k d n")  # P.T is [_ _ 3 N]
        """
        dst is the target points, Q is our problem definition from above and w are the parameters of the probkem that we wish to find
        Qw = dst
        w = Q^-1*dst
        """
        dst = th.zeros((bs, K, N + d + 1, d), dtype=dtype).to(device)
        dst[:, :, :N, :d] = destination_pts

        # time to solve your problem.
        # make sure Q_mat is invertible
        Q_mat = Q_mat + th.eye(N + 3, device=device)[None, None, :, :] * _eps
        Q_inv = th.inverse(Q_mat)
        W = th.einsum("bknm,bkmd->bknd", Q_inv, dst)

        theta = rearrange(
            W[:, :, N:, :], "b K s d -> b K d s"
        )  # we want theta to return affine transofrmation per dimension
        control_params = W[:, :, :N, :]
        control_points = source_pts

        def warp_func(
            coordinates: th.Tensor,
            theta: th.Tensor,
            control_params: th.Tensor,
            control_points: th.Tensor,
        ) -> th.Tensor:

            """
            :param: coordinates - `[bs, h, w, d]`
            :param: thetha - `[bs, K, d, d+1]` those are params for an affined transform
            :param: control_param - `[bs, K, N, d]`

            :return: `[bs, K, h, w, d]`
            """

            d = coordinates.shape[-1]

            # [b K h w d] + [bs K 1 1 d]
            transformed_cords = (
                th.einsum("bKdv,bhwv->bKhwd", theta[:, :, :, :d], coordinates)
                + theta[:, :, None, None, :, d]
            )
            # [b h w d] -> [b 1 1 h w d]
            _cords = coordinates[:, None, None, :, :, :]
            # [bs K N d] -> [b K N 1 1 d]
            _ctrl_pts = control_points[:, :, :, None, None, :]
            # [bs K N h w]
            dists_sq = fused_dist_square(_cords, _ctrl_pts)
            basis_vals = fused_basis_func(dists_sq)

            jitter = th.einsum("bKNhw,bKNd->bKhwd", basis_vals, control_params)
            # jitter has a N in its dimension. looks like i should sum over it but the
            # original impl. doesnt do it.
            out = transformed_cords + jitter

            return out

        inst = cls(
            theta=theta,
            control_params=control_params,
            control_points=control_points,
            _warp_cords=warp_func,
        )

        inst._warp_cords.__doc__ = warp_func.__doc__
        return inst

