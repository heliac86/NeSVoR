from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .hash_grid_torch import HashEmbedder
from ..transform import RigidTransform, ax_transform_points, mat_transform_points
from ..utils import resolution2sigma
# ===== [FF Loss 추가] losses.py에서 FocalFrequencyLoss 및 FF_LOSS 키 가져오기 =====
from .losses import FocalFrequencyLoss, FF_LOSS
# ===== [FF Loss 추가 끝] =====

USE_TORCH = False

if not USE_TORCH:
    try:
        import tinycudann as tcnn
    except:
        logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
        USE_TORCH = True


# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
B_REG = "biasReg"
T_REG = "transReg"
I_REG = "imageReg"
D_REG = "deformReg"


def build_encoding(**config):
    if USE_TORCH:
        encoding = HashEmbedder(**config)
    else:
        n_input_dims = config.pop("n_input_dims")
        dtype = config.pop("dtype")
        try:
            encoding = tcnn.Encoding(
                n_input_dims=n_input_dims, encoding_config=config, dtype=dtype
            )
        except RuntimeError as e:
            if "TCNN was not compiled with half-precision support" in str(e):
                logging.error(
                    "TCNN was not compiled with half-precision support! "
                    "Try using --single-precision in the nesvor command! "
                )
            raise e
    return encoding


def build_network(**config):
    dtype = config.pop("dtype")
    assert dtype == torch.float16 or dtype == torch.float32
    if dtype == torch.float16 and not USE_TORCH:
        return tcnn.Network(
            n_input_dims=config["n_input_dims"],
            n_output_dims=config["n_output_dims"],
            network_config={
                "otype": "CutlassMLP",
                "activation": config["activation"],
                "output_activation": config["output_activation"],
                "n_neurons": config["n_neurons"],
                "n_hidden_layers": config["n_hidden_layers"],
            },
        )
    else:
        activation = (
            None
            if config["activation"] == "None"
            else getattr(nn, config["activation"])
        )
        output_activation = (
            None
            if config["output_activation"] == "None"
            else getattr(nn, config["output_activation"])
        )
        models = []
        if config["n_hidden_layers"] > 0:
            models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))
            for _ in range(config["n_hidden_layers"] - 1):
                if activation is not None:
                    models.append(activation())
                models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))
            if activation is not None:
                models.append(activation())
            models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
        else:
            models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))
        if output_activation is not None:
            models.append(output_activation())
        return nn.Sequential(*models)


def compute_resolution_nlevel(
    bounding_box: torch.Tensor,
    coarsest_resolution: float,
    finest_resolution: float,
    level_scale: float,
    spatial_scaling: float,
) -> Tuple[int, int]:
    base_resolution = (
        (
            (bounding_box[1] - bounding_box[0]).max()
            * spatial_scaling
            / coarsest_resolution
        )
        .ceil()
        .int()
        .item()
    )
    n_levels = (
        (
            torch.log2(
                (bounding_box[1] - bounding_box[0]).max()
                * spatial_scaling
                / finest_resolution
                / base_resolution
            )
            / log2(level_scale)
            + 1
        )
        .ceil()
        .int()
        .item()
    )
    return int(base_resolution), int(n_levels)


class INR(nn.Module):
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)
        # hash grid encoding
        base_resolution, n_levels = compute_resolution_nlevel(
            self.bounding_box,
            args.coarsest_resolution,
            args.finest_resolution,
            args.level_scale,
            spatial_scaling,
        )

        # ===== [k_norm] 역정규화 스케일 인자 =====
        # train.py에서 훈련 직전 dataset.mean 값으로 덮어씀.
        # 1.0으로 초기화하면 정규화를 적용하지 않은 기존 동작과 동일하게 작동함
        # (호환성 유지 및 롤백 시 이 두 줄 삭제)
        self.register_buffer("v_mean", torch.tensor(1.0, dtype=torch.float32))
        # ===== [k_norm 끝] =====

        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=args.level_scale,
            dtype=args.dtype,
        )
        # density net
        self.density_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level,
            n_output_dims=1 + args.n_features_z,
            activation="ReLU",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=args.depth,
            dtype=torch.float32 if args.img_reg_autodiff else args.dtype,
        )

        # ===== [복구] Gating 연산과 차원 변환에 필수적인 변수 선언 =====
        self.n_levels = n_levels
        self.n_features_per_level = args.n_features_per_level
        # ==========================================================

        # [추가] 공간 인지형 Gating Network (Spatial Gating MLP)=======
        # x, y, z (3차원)을 입력받아 n_levels 개의 0~1 사이 가중치를 출력
        self.gating_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_levels),
            nn.Sigmoid()
        )
        
        # 페널티 값을 저장해둘 임시 변수
        self.gating_penalty = 0.0
        # ============================================================

        # logging
        logging.debug(
            "hyperparameters for hash grid encoding: "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * args.level_scale ** (n_levels - 1)),
            args.level_scale,
            n_levels,
        )
        logging.debug(
            "bounding box for reconstruction (mm): "
            + "x=[%f, %f], y=[%f, %f], z=[%f, %f]",
            self.bounding_box[0, 0],
            self.bounding_box[1, 0],
            self.bounding_box[0, 1],
            self.bounding_box[1, 1],
            self.bounding_box[0, 2],
            self.bounding_box[1, 2],
        )

    def forward(self, x: torch.Tensor):
        x_norm = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0]) # 수정
        prefix_shape = x.shape[:-1]
        x_flat = x_norm.view(-1, x_norm.shape[-1]) # 수정
        pe = self.encoding(x_flat)
        if not self.training:
            pe = pe.to(dtype=x_flat.dtype)

        # ===== [추가] Spatial Gating 적용 (수정: 저주파 면세 및 강제 개방) =====
        # 1. 위치 정보(x_flat)를 MLP에 넣어 공간 가중치 생성 (Shape: N, n_levels)
        raw_spatial_weights = self.gating_net(x_flat)
        
        # [핵심 개조] 저주파 구간(Level 0 ~ 3) 강제 1.0 고정
        n_base_levels = 4 # 0, 1, 2, 3번 레벨은 뼈대(거시적 형태)를 담당
        
        # 앞의 4개 레벨은 무조건 1.0(완전 개방)으로 덮어씌움
        base_weights = torch.ones_like(raw_spatial_weights[:, :n_base_levels])
        # 뒤의 8개 레벨은 MLP가 판단한 가중치 그대로 사용
        high_weights = raw_spatial_weights[:, n_base_levels:]
        
        # 두 텐서를 결합하여 최종 가중치 완성
        spatial_weights = torch.cat([base_weights, high_weights], dim=-1)
        
        # 2. 페널티(세금) 계산 및 내부 저장 (저주파는 면세, 고주파에만 과세)
        if self.training:
            # 고주파 레벨(4~11)에 대해서만 차등 페널티 가중치 생성 (예: 0.1부터 2.0까지)
            penalty_factors = torch.linspace(0.1, 2.0, steps=(self.n_levels - n_base_levels), device=spatial_weights.device)
            
            # 면세 구간(base_weights)은 놔두고, 고주파(high_weights)에만 페널티 부과
            weighted_penalty = high_weights.abs() * penalty_factors
            self.gating_penalty = weighted_penalty.mean()

            # ===== [관찰용 디버그 코드] =====
            # 속도 저하를 막기 위해 내부 카운터를 만들어 500번에 1번씩만 평균값 출력
            if not hasattr(self, '_debug_step'):
                self._debug_step = 0
            self._debug_step += 1
            
            if self._debug_step % 500 == 0:
                # N개의 픽셀에 대한 레벨별 평균을 구하고, 보기 좋게 소수점 3자리로 출력
                mean_w = spatial_weights.mean(dim=0).detach().cpu().numpy()
                import numpy as np
                print(f"\n[Step {self._debug_step}] Level-wise Mean Gating: {np.round(mean_w, 3)}")
            # ===== [디버그 코드 끝] =====

        # 3. 해시 그리드 특징(pe)에 가중치 곱하기
        pe = pe.view(-1, self.n_levels, self.n_features_per_level)
        pe = pe * spatial_weights.unsqueeze(-1)  # (N, L, F) * (N, L, 1)
        pe = pe.view(-1, self.n_levels * self.n_features_per_level)
        # ===== [추가 끝] =====
        
        z = self.density_net(pe)
        density = F.softplus(z[..., 0].view(prefix_shape))
        if self.training:
            return density, pe, z
        else:
            return density

    def sample_batch(
        self,
        xyz: torch.Tensor,
        transformation: Optional[RigidTransform],
        psf_sigma: Union[float, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples > 1:
            if isinstance(psf_sigma, torch.Tensor):
                psf_sigma = psf_sigma.view(-1, 1, 3)
            xyz_psf = torch.randn(
                xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
            )
            xyz = xyz[:, None] + xyz_psf * psf_sigma
        else:
            xyz = xyz[:, None]
        if transformation is not None:
            trans_first = transformation.trans_first
            mat = transformation.matrix(trans_first)
            xyz = mat_transform_points(mat[:, None], xyz, trans_first)
        return xyz


class DeformNet(nn.Module):
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)
        # hash grid encoding
        base_resolution, n_levels = compute_resolution_nlevel(
            bounding_box,
            args.coarsest_resolution_deform,
            args.finest_resolution_deform,
            args.level_scale_deform,
            spatial_scaling,
        )
        level_scale = args.level_scale_deform

        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level_deform,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=level_scale,
            dtype=args.dtype,
            interpolation="Smoothstep",
        )
        self.deform_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level_deform
            + args.n_features_deform,
            n_output_dims=3,
            activation="Tanh",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=2,
            dtype=torch.float32,
        )
        for p in self.deform_net.parameters():
            torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)
        logging.debug(
            "hyperparameters for hash grid encoding (deform net): "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * level_scale ** (n_levels - 1)),
            level_scale,
            n_levels,
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        pe = self.encoding(x)
        inputs = torch.cat((pe, e.reshape(-1, e.shape[-1])), -1)
        outputs = self.deform_net(inputs) + x
        outputs = (
            outputs * (self.bounding_box[1] - self.bounding_box[0])
            + self.bounding_box[0]
        )
        return outputs.view(x_shape)


class NeSVoR(nn.Module):
    def __init__(
        self,
        transformation: RigidTransform,
        resolution: torch.Tensor,
        v_mean: float,
        bounding_box: torch.Tensor,
        spatial_scaling: float,
        args: Namespace,
    ) -> None:
        super().__init__()
        if "cpu" in str(args.device):  # CPU mode
            global USE_TORCH
            USE_TORCH = True
        else:
            # set default GPU for tinycudann
            torch.cuda.set_device(args.device)
        self.spatial_scaling = spatial_scaling
        self.args = args
        self.n_slices = 0
        self.trans_first = True
        self.transformation = transformation
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)
        self.delta = args.delta * v_mean
        self.build_network(bounding_box)
        self.to(args.device)

    @property
    def transformation(self) -> RigidTransform:
        return RigidTransform(self.axisangle.detach(), self.trans_first)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)
        axisangle = value.axisangle(self.trans_first)
        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor
        self.register_buffer("axisangle_init", axisangle.detach().clone())
        if not self.args.no_transformation_optimization:
            self.axisangle = nn.Parameter(axisangle.detach().clone())
        else:
            self.register_buffer("axisangle", axisangle.detach().clone())

    def build_network(self, bounding_box) -> None:
        if self.args.n_features_slice:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )
        if not self.args.no_slice_scale:
            self.logit_coef = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        if not self.args.no_slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        if self.args.deformable:
            self.deform_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_deform
            )
            self.deform_net = DeformNet(bounding_box, self.args, self.spatial_scaling)
        # INR
        self.inr = INR(bounding_box, self.args, self.spatial_scaling)
        # sigma net
        if not self.args.no_pixel_variance:
            self.sigma_net = build_network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )
        # bias net
        if self.args.n_levels_bias:
            self.b_net = build_network(
                n_input_dims=self.args.n_levels_bias * self.args.n_features_per_level
                + self.args.n_features_slice,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )
        # ===== [FF Loss 추가] FocalFrequencyLoss 인스턴스 초기화 =====
        # weight_ff_loss > 0일 때만 생성. args에 해당 필드가 없으면 0으로 간주 (parsers.py 추가 전 호환성)
        if getattr(self.args, "weight_ff_loss", 0.0) > 0:
            self.ff_loss_fn = FocalFrequencyLoss(
                alpha=getattr(self.args, "ff_alpha", 1.0),
                patch_factor=1,
                freq_mask_ratio=getattr(self.args, "ff_mask_ratio", 1.0)
            )
        else:
            self.ff_loss_fn = None
        # ===== [FF Loss 추가 끝] =====

    def forward(
        self,
        xyz: torch.Tensor,
        v: torch.Tensor,
        slice_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        # sample psf point
        batch_size = xyz.shape[0]
        n_samples = self.args.n_samples
        xyz_psf = torch.randn(
            batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
        )
        # psf = 1
        psf_sigma = self.psf_sigma[slice_idx][:, None]
        # transform points
        t = self.axisangle[slice_idx][:, None]
        xyz = ax_transform_points(
            t, xyz[:, None] + xyz_psf * psf_sigma, self.trans_first
        )

        # deform
        xyz_ori = xyz
        if self.args.deformable:
            de = self.deform_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
            xyz = self.deform_net(xyz, de)

        # inputs
        if self.args.n_features_slice:
            se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
        else:
            se = None
        # forward
        results = self.net_forward(xyz, se)
        # output
        density = results["density"]
        if "log_bias" in results:
            log_bias = results["log_bias"]
            bias = log_bias.exp()
            bias_detach = bias.detach()
        else:
            log_bias = 0
            bias = 1
            bias_detach = 1
        if "log_var" in results:
            log_var = results["log_var"]
            var = log_var.exp()
        else:
            log_var = 0
            var = 1
        # imaging
        if not self.args.no_slice_scale:
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
        else:
            c = 1
        v_out = (bias * density).mean(-1)
        v_out = c * v_out
        if not self.args.no_pixel_variance:
            # var = (bias_detach * psf * var).mean(-1)
            var = (bias_detach * var).mean(-1)
            # bugfix: --no-slice-scale 시 c=1(int)이므로 tensor 여부를 확인한 뒤 detach
            c_detach = c.detach() if isinstance(c, torch.Tensor) else c
            var = c_detach * var
            var = var**2
        if not self.args.no_slice_variance:
            var = var + self.log_var_slice.exp()[slice_idx]
        # losses
        losses = {D_LOSS: ((v_out - v) ** 2 / (2 * var)).mean()}

        if not (self.args.no_pixel_variance and self.args.no_slice_variance):
            losses[S_LOSS] = 0.5 * var.log().mean()
            losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]
        if not self.args.no_transformation_optimization:
            losses[T_REG] = self.trans_loss(trans_first=self.trans_first)
        if self.args.n_levels_bias:
            losses[B_REG] = log_bias.mean() ** 2
        if self.args.deformable:
            losses[D_REG] = self.deform_reg(
                xyz, xyz_ori, de
            )  # deform_reg_autodiff(self.deform_net, xyz_ori, de)
        # image regularization
        losses[I_REG] = self.img_reg(density, xyz)

        # ===== [추가] Gating Sparsity Loss 수집 =====
        # INR.forward에서 계산해둔 페널티를 가져옵니다.
        if hasattr(self.inr, 'gating_penalty') and self.inr.training:
            losses["gatingReg"] = self.inr.gating_penalty
        # ===== [추가 끝] =====

        return losses

    # ===== [FF Loss 추가] 패치 단위 순전파 및 FF Loss 계산 메서드 =====
    def patch_forward(
        self,
        xyz_patch: torch.Tensor,
        v_patch: torch.Tensor,
        slice_idx_patch: torch.Tensor,
        valid_mask_patch: torch.Tensor,
    ) -> Dict[str, Any]:
        n = xyz_patch.shape[0]
        PP = xyz_patch.shape[1]   # P*P
        P = int(PP ** 0.5)

        xyz_flat = xyz_patch.view(n * PP, 3)
        slice_idx_flat = slice_idx_patch.repeat_interleave(PP)  # (n*P*P,)

        n_samples = self.args.n_samples
        xyz_psf = torch.randn(
            n * PP, n_samples, 3, dtype=xyz_flat.dtype, device=xyz_flat.device
        )
        psf_sigma = self.psf_sigma[slice_idx_flat][:, None]
        t = self.axisangle[slice_idx_flat][:, None]
        xyz_t = ax_transform_points(
            t, xyz_flat[:, None] + xyz_psf * psf_sigma, self.trans_first
        )

        if self.args.deformable:
            de = self.deform_embedding(slice_idx_flat)[:, None].expand(
                -1, n_samples, -1
            )
            xyz_t = self.deform_net(xyz_t, de)

        if self.args.n_features_slice:
            se = self.slice_embedding(slice_idx_flat)[:, None].expand(
                -1, n_samples, -1
            )
        else:
            se = None

        results = self.net_forward(xyz_t, se)
        density = results["density"]

        if "log_bias" in results:
            bias = results["log_bias"].exp()
        else:
            bias = 1

        if not self.args.no_slice_scale:
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx_flat] * self.n_slices
        else:
            c = 1

        v_pred = (bias * density).mean(-1)
        v_pred = c * v_pred
        v_pred_patch = v_pred.view(n, P, P)

        mask_float = valid_mask_patch.float()
        v_pred_patch = v_pred_patch * mask_float
        v_gt_patch   = v_patch * mask_float

        ff_loss = self.ff_loss_fn(v_pred_patch, v_gt_patch)

        return {FF_LOSS: ff_loss}
    # ===== [FF Loss 추가 끝] =====

    def net_forward(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        density, pe, z = self.inr(x)
        prefix_shape = density.shape
        results = {"density": density}

        zs = []
        if se is not None:
            zs.append(se.reshape(-1, se.shape[-1]))

        if self.args.n_levels_bias:
            pe_bias = pe[
                ..., : self.args.n_levels_bias * self.args.n_features_per_level
            ]
            results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
                prefix_shape
            )

        if not self.args.no_pixel_variance:
            zs.append(z[..., 1:])
            results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

        return results

    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        x = RigidTransform(self.axisangle, trans_first=trans_first)
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)
        err = y.inv().compose(x).axisangle(trans_first=trans_first)
        loss_R = torch.mean(err[:, :3] ** 2)
        loss_T = torch.mean(err[:, 3:] ** 2)
        return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T

    def img_reg(self, density, xyz):
        if self.args.image_regularization == "none":
            return torch.zeros((1,), dtype=density.dtype, device=density.device)

        if self.args.img_reg_autodiff:
            n_sample = 4
            xyz = xyz[:, :n_sample].flatten(0, 1).detach()
            xyz.requires_grad_()
            density, _, _ = self.inr(xyz)
            grad = (
                torch.autograd.grad((density.sum(),), (xyz,), create_graph=True)[0]
                / self.spatial_scaling
            )
            grad2 = grad.pow(2)
        else:
            xyz = xyz * self.spatial_scaling
            d_density = density - torch.flip(density, (1,))
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
            grad2 = d_density**2 / dx2

        if self.args.image_regularization == "TV":
            return grad2.sqrt().mean()
        elif self.args.image_regularization == "edge":
            return self.delta * (
                (1 + grad2 / (self.delta * self.delta)).sqrt().mean() - 1
            )
        elif self.args.image_regularization == "L2":
            return grad2.mean()
        else:
            raise ValueError("unknown image regularization!")

    def deform_reg(self, out, xyz, e):
        if True:  # use autodiff
            n_sample = 4
            x = xyz[:, :n_sample].flatten(0, 1).detach()
            e = e[:, :n_sample].flatten(0, 1).detach()

            x.requires_grad_()
            outputs = self.deform_net(x, e)
            grads = []
            out_sum = []
            for i in range(3):
                out_sum.append(outputs[:, i].sum())
                grads.append(
                    torch.autograd.grad((out_sum[-1],), (x,), create_graph=True)[0]
                )
            jacobian = torch.stack(grads, -1)
            jtj = torch.matmul(jacobian, jacobian.transpose(-1, -2))
            I = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
            sq_residual = ((jtj - I) ** 2).sum((-2, -1))
            return torch.nan_to_num(sq_residual, 0.0, 0.0, 0.0).mean()
        else:
            out = out - xyz
            d_out2 = ((out - torch.flip(out, (1,))) ** 2).sum(-1) + 1e-6
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
            dd_dx = d_out2.sqrt() / dx2.sqrt()
            return F.smooth_l1_loss(dd_dx, torch.zeros_like(dd_dx).detach(), beta=1e-3)
