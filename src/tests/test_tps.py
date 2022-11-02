import pytest
import os
import sys
from pathlib import Path

pkg_path = str(Path(__file__).parent.parent)
sys.path.append(pkg_path)

import torch
from einops import rearrange
from transmotion.cord_warp import ThinPlateSpline, make_coordinate_grid


class _OrigTPSComp:
    """
    This is a copy from the original code base, used to validate that my implementation is the same
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.
    """

    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == "random":
            noise = torch.normal(
                mean=0, std=kwargs["sigma_affine"] * torch.ones([bs, 2, 3])
            )
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid(
                # (kwargs["points_tps"], kwargs["points_tps"]), type=noise.type()
                # Differ from the original, we use dtype
                kwargs["points_tps"],
                kwargs["points_tps"],
                dtype=noise.dtype,
            )
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0,
                std=kwargs["sigma_tps"]
                * torch.ones([bs, 1, kwargs["points_tps"] ** 2]),
            )
        elif mode == "kp":
            kp_1 = kwargs["kp_1"]  # [bs, K, N, 2]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]  # K
            """
              ...
            [x_n y_n 1]
            [0 0 0]
            [0 0 0]
            [0 0 0]            

            [ d_1 ... d_N ] | [x_1 y_1 1]
            [ d_1 ... d_N ] | . ..
            [ d_1 ... d_N ] | [x_N y_N 1]
            [x_1  ...  x_N] | 
            [y_1       y_N] | .   0
            [1           1] |

            """

            n = kp_1.shape[2]  # N
            K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
            # K has the following shape: [bs, K, N, 1, 2] - [bs, K ,1, N, 2] -> [b, K, N, N, 2] -> [b, K, N, N]
            K = K**2
            K = K * torch.log(K + 1e-9)  # [b, K, N, N]
            one1 = (
                torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1)
                .to(device)
                .type(kp_type)
            )
            kp_1p = torch.cat([kp_1, one1], 3)  # [b K N 3]

            # b K 3 3
            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            # [b K N+3 3]
            P = torch.cat([kp_1p, zero], 2)
            # [bs K N N] | [b K 3 N] => [bs K N+3 N]
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            # [b K N N] [b K 3 N] -> [b K N+3 N]
            L = torch.cat([L, P], 3)
            # [b K N+3 N] [b K N+3 3] -> [b K N+3 N+3]

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            # bs K 3 2
            Y = torch.cat([kp_2, zero], 2)
            # b K N+3 2
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type) * 0.01
            # [1 1 N+3 N+3]
            L = L + one
            # ^^ this was to make it invertable

            param = torch.matmul(torch.inverse(L), Y)
            # [b K N+3 N+3] x [b K N+3 2] -> [b K N+3 2]
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)
            # [b K 2 3]

            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
            # [b K N 2]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = (
            make_coordinate_grid(*frame.shape[2:], dtype=frame.dtype)
            .unsqueeze(0)
            .to(frame.device)
        )
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == "kp":
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        # [bs, K, 2, 3]
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        # [bs, K, N, 2]
        control_points = self.control_points.type(coordinates.type()).to(
            coordinates.device
        )
        # [bs, K, N, 2]
        control_params = self.control_params.type(coordinates.type()).to(
            coordinates.device
        )

        if self.mode == "kp":
            transformed = (
                torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1))
                + theta[:, :, :, 2:]
            )
            # [bs, K, N P, 2]
            distances = coordinates.view(
                # [1 1 1 P 2]
                coordinates.shape[0],
                1,
                1,
                -1,
                2
                # [bs K N 1 2]
            ) - control_points.view(self.bs, control_points.shape[1], -1, 1, 2)
            # -> [bs K N P 2]
            distances = distances**2
            result = distances.sum(-1)
            # [bs K N P]
            result = result * torch.log(result + 1e-9)
            # [bs K P N] x [bs K N 2] => [bs K P 2]
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            # [bs K 2 P] -permute-> [bs K P 2] + [bs K P 2] -> [bs K P 2]
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == "random":
            # coordintaes will have [bs, P, 2]
            theta = theta.unsqueeze(1)
            # [bs 1 2 2] x [bs, P, 2, 1] -> [bs P 2 1] -squeez-> [b P 2]
            transformed = (
                torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1))
                + theta[:, :, :, 2:]
            )
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(
                coordinates.shape[0], -1, 1, 2
            ) - control_points.view(1, 1, -1, 2)
            # [bs P 1 2] - [1 1 K*N 2] -> [bs P K*N 2]
            distances = ances**2

            result = distances.sum(-1)
            # [bs P K*N]
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            # [bs KN P] * [bs 1 P]
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
            # [bs KN P] -> maybe K hides the xy coordinates and K is actually 2?
        else:
            raise Exception("Error TPS mode")

        return transformed


def _random_kp(bs=16, k=10, n=5):
    kp_1 = torch.randn((bs, k, n, 2))
    kp_2 = torch.randn((bs, k, n, 2))
    return kp_1, kp_2


def test_fit():
    bs = 16
    kp_1, kp_2 = _random_kp(bs=bs)

    grid = make_coordinate_grid(32, 32)
    grid = grid[None, ...]
    my_tps = ThinPlateSpline.fit(kp_1, kp_2)
    my_grid = my_tps.warp_cords(grid)

    tps = _OrigTPSComp("kp", bs, kp_1=kp_1, kp_2=kp_2)
    grid = rearrange(grid, "b h w d -> b (h w) d", b=1)
    tps_grid = tps.warp_coordinates(grid)
    comp_grid = rearrange(my_grid, "b k h w d -> b k (h w) d")

    assert torch.allclose(
        comp_grid, tps_grid
    ), "new thin plate and original are not the same"


@pytest.mark.skip("Need to re-impl random to match the generated random weights")
def test_random():
    """New random impl doesn't allow to pass"""
    bs = 16
    points_tps = 5
    sigma_affine = 0.001
    sigma_tps = sigma_affine

    grid = make_coordinate_grid(32, 32)
    grid = grid[None, ...]
    my_tps = ThinPlateSpline.random(0.001, 5, batch_size=bs, sigma_affine=sigma_affine)
    my_grid = my_tps.warp_cords(grid)

    tps = _OrigTPSComp(
        "random",
        bs,
        points_tps=points_tps,
        sigma_tps=sigma_tps,
        sigma_affine=sigma_affine,
    )
    grid = rearrange(grid, "b h w d -> b (h w) d")
    tps_grid = tps.warp_coordinates(grid)
    comp_grid = rearrange(my_grid, "b k h w d -> b k (h w) d")

    assert torch.allclose(
        comp_grid.squeeze(), tps_grid
    ), "new thin plate and original are not the same"
