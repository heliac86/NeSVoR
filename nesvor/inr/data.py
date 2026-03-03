from typing import Dict, List
import torch
from ..utils import gaussian_blur
from ..transform import RigidTransform, transform_points
from ..image import Volume, Slice


class PointDataset(object):
    def __init__(self, slices: List[Slice]) -> None:
        self.mask_threshold = 1  # args.mask_threshold

        xyz_all = []
        v_all = []
        slice_idx_all = []
        transformation_all = []
        resolution_all = []

        # ===== [FF Loss 추가] 패치 샘플링을 위한 슬라이스별 2D 데이터 보존 =====
        self.slice_images: List[torch.Tensor] = []        # 슬라이스별 (H, W) 이미지
        self.slice_masks: List[torch.Tensor] = []         # 슬라이스별 (H, W) bool 마스크
        self.slice_shape_xyz: List[torch.Tensor] = []     # 슬라이스별 shape_xyz [W, H, 1]
        self.slice_resolution_xyz: List[torch.Tensor] = []  # 슬라이스별 resolution_xyz
        # ===== [FF Loss 추가 끝] =====

        for i, slice in enumerate(slices):
            xyz = slice.xyz_masked_untransformed
            v = slice.v_masked
            slice_idx = torch.full(v.shape, i, device=v.device)
            xyz_all.append(xyz)
            v_all.append(v)
            slice_idx_all.append(slice_idx)
            transformation_all.append(slice.transformation)
            resolution_all.append(slice.resolution_xyz)

            # ===== [FF Loss 추가] 슬라이스 2D 데이터 저장 =====
            self.slice_images.append(slice.image[0].detach().clone())      # (H, W)
            self.slice_masks.append(slice.mask[0].detach().clone())        # (H, W)
            self.slice_shape_xyz.append(slice.shape_xyz.clone())           # [W, H, 1]
            self.slice_resolution_xyz.append(slice.resolution_xyz.clone()) # [rx, ry, rz]
            # ===== [FF Loss 추가 끝] =====

        self.xyz = torch.cat(xyz_all)
        self.v = torch.cat(v_all)
        self.slice_idx = torch.cat(slice_idx_all)
        self.transformation = RigidTransform.cat(transformation_all)
        self.resolution = torch.stack(resolution_all, 0)
        self.count = self.v.shape[0]
        self.epoch = 0

    @property
    def bounding_box(self) -> torch.Tensor:
        max_r = self.resolution.max()
        xyz_transformed = self.xyz_transformed
        xyz_min = xyz_transformed.amin(0) - 2 * max_r
        xyz_max = xyz_transformed.amax(0) + 2 * max_r
        bounding_box = torch.stack([xyz_min, xyz_max], 0)
        return bounding_box

    @property
    def mean(self) -> float:
        q1, q2 = torch.quantile(
            self.v if self.v.numel() < 256 * 256 * 256 else self.v[: 256 * 256 * 256],
            torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
        )
        return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

    # ===== [k_norm] 훈련 타겟 일괄 정규화 =====
    # self.v와 self.slice_images를 v_mean으로 동시에 나누어
    # MSE Loss와 FF Loss의 타겟 스케일을 일치시킴.
    # 롤백 시이 메서드를 삭제하면 정규화가 해제됨.
    def normalize(self, v_mean: float) -> None:
        """v_mean으로 self.v 및 self.slice_images를 함께 나누어 정규화.

        train.py에서 dataset.mean을 쪼은 뒤 한 번만 호출.
        정규화 후 모델 출력은 ~1.0 단위이며,
        sample.py에서 * model.v_mean 역정규화를 통해 원본 강도 범위로 복원됨.
        """
        if v_mean <= 0:
            raise ValueError(f"v_mean must be positive, got {v_mean}")
        self.v = self.v / v_mean
        self.slice_images = [
            img / v_mean for img in self.slice_images
        ]
    # ===== [k_norm 끝] =====

    def get_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        if self.count + batch_size > self.xyz.shape[0]:  # new epoch, shuffle data
            self.count = 0
            self.epoch += 1
            idx = torch.randperm(self.xyz.shape[0], device=device)
            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]
        # fetch a batch of data
        batch = {
            "xyz": self.xyz[self.count : self.count + batch_size],
            "v": self.v[self.count : self.count + batch_size],
            "slice_idx": self.slice_idx[self.count : self.count + batch_size],
        }
        self.count += batch_size
        return batch

    # ===== [FF Loss 추가] 패치 단위 샘플링 메서드 =====
    def get_patch_batch(
        self,
        n_patches: int,
        patch_size: int,
        device,
    ) -> Dict[str, torch.Tensor]:
        """FF Loss 계산을 위한 패치 배치 샘플링.

        get_batch()는 픽셀을 랜덤으로 섯어 공간 구조가 파괴되지만,
        이 메서드는 한 슬라이스에서 공간적으로 연속된 P×P 패치를 추출하므로
        FFT 적용 및 FF Loss 계산에 적합하다.

        좌표 계산 방식은 Image.xyz_masked_untransformed 의 공식과 동일:
            coord = (kji - (shape_xyz - 1) / 2) * resolution_xyz
            (kji 순서: [col(x), row(y), depth(z)])

        Args:
            n_patches (int): 샘플링할 패치 수.
            patch_size (int): 패치의 한 변 크기 P (P×P 패치).
            device: 출력 텐서를 배치할 디바이스.

        Returns:
            dict:
                "xyz_patch"        : (n, P*P, 3)  untransformed 3D 좌표
                "v_patch"          : (n, P, P)    GT 픽셀 강도값
                "slice_idx_patch"  : (n,)         패치가 속한 슬라이스 인덱스
                "valid_mask_patch" : (n, P, P)    유효 픽셀 bool 마스크
            패치 크기보다 작은 슬라이스만 존재하면 빈 dict를 반환한다.
        """
        P = patch_size
        n_slices = len(self.slice_images)

        xyz_list = []
        v_list = []
        sidx_list = []
        valid_list = []

        for _ in range(n_patches):
            # 랜덤 슬라이스 선택
            s_idx = int(torch.randint(n_slices, (1,)).item())
            img = self.slice_images[s_idx]            # (H, W)
            msk = self.slice_masks[s_idx]             # (H, W)
            shape_xyz = self.slice_shape_xyz[s_idx]   # [W, H, 1]
            res_xyz = self.slice_resolution_xyz[s_idx]  # [rx, ry, rz]

            H, W = img.shape

            # 패치보다 작은 슬라이스는 건너뜀
            if H < P or W < P:
                continue

            # 랜덤 패치 좌표 (좌상단 기준)
            r0 = int(torch.randint(0, H - P + 1, (1,)).item())
            c0 = int(torch.randint(0, W - P + 1, (1,)).item())

            # GT 강도값 및 유효 마스크 추출
            v_patch = img[r0 : r0 + P, c0 : c0 + P].to(device)       # (P, P)
            valid_patch = msk[r0 : r0 + P, c0 : c0 + P].to(device)   # (P, P)

            # 패치 내 각 픽셀의 untransformed 3D 좌표 계산
            # Image.xyz_masked_untransformed 공식과 동일:
            #   kji = [col(x), row(y), depth(z)]
            #   coord = (kji - (shape_xyz - 1) / 2) * resolution_xyz
            rows = torch.arange(r0, r0 + P, dtype=torch.float32, device=device)  # (P,)
            cols = torch.arange(c0, c0 + P, dtype=torch.float32, device=device)  # (P,)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")        # (P, P)
            z_grid = torch.zeros_like(row_grid)                                   # (P, P)

            # kji: [x=col, y=row, z=0], shape (P, P, 3)
            kji = torch.stack([col_grid, row_grid, z_grid], dim=-1)

            shape_xyz_dev = shape_xyz.to(device).float()
            res_dev = res_xyz.to(device).float()
            xyz_patch = (kji - (shape_xyz_dev - 1) / 2) * res_dev  # (P, P, 3)

            xyz_list.append(xyz_patch.view(P * P, 3))   # (P*P, 3)
            v_list.append(v_patch)                      # (P, P)
            sidx_list.append(
                torch.full((1,), s_idx, dtype=torch.long, device=device)
            )
            valid_list.append(valid_patch)              # (P, P)

        if len(xyz_list) == 0:
            return {}

        return {
            "xyz_patch":         torch.stack(xyz_list, dim=0),           # (n, P*P, 3)
            "v_patch":           torch.stack(v_list, dim=0),             # (n, P, P)
            "slice_idx_patch":   torch.cat(sidx_list, dim=0),            # (n,)
            "valid_mask_patch":  torch.stack(valid_list, dim=0),         # (n, P, P)
        }
    # ===== [FF Loss 추가 끝] =====

    @property
    def xyz_transformed(self) -> torch.Tensor:
        return transform_points(self.transformation[self.slice_idx], self.xyz)

    @property
    def mask(self) -> Volume:
        with torch.no_grad():
            resolution_min = self.resolution.min()
            resolution_max = self.resolution.max()
            xyz = self.xyz_transformed
            xyz_min = xyz.amin(0) - resolution_max * 10
            xyz_max = xyz.amax(0) + resolution_max * 10
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
            shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
            kji = ((xyz - xyz_min) / resolution_min).round().long()

            mask = torch.bincount(
                kji[..., 0]
                + shape[2] * kji[..., 1]
                + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            )
            mask = mask.view((1, 1) + shape).float()
            mask_threshold = (
                self.mask_threshold
                * resolution_min**3
                / self.resolution.log().mean().exp() ** 3
            )
            mask_threshold *= mask.sum() / (mask > 0).sum()
            assert len(mask.shape) == 5
            mask = (
                gaussian_blur(mask, (resolution_max / resolution_min).item(), 3)
                > mask_threshold
            )[0, 0]

            xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min
            return Volume(
                mask.float(),
                mask,
                RigidTransform(torch.cat([0 * xyz_c, xyz_c])[None], True),
                resolution_min,
                resolution_min,
                resolution_min,
            )
