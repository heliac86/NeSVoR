"""
losses.py
---------
Focal Frequency Loss (FFL) 구현 모듈.

참고 논문:
  Focal Frequency Loss for Image Reconstruction and Synthesis
  Lama et al., ICCV 2021
  https://arxiv.org/abs/2012.12821

NeSVoR 학습 루프에서 패치 단위 FF Loss로 사용된다.
models.py 에서 FF_LOSS 키로 참조하고,
train.py 의 loss_weights 딕셔너리에 가중치를 등록한다.
"""

import torch
import torch.nn as nn


# models.py 의 D_LOSS, S_LOSS 등과 동일한 네이밍 규칙
FF_LOSS = "FFLoss"


class FocalFrequencyLoss(nn.Module):
    """패치 단위 Focal Frequency Loss.

    예측 패치와 정답 패치를 2D FFT로 주파수 공간으로 변환한 뒤,
    재현하기 어려운(hard) 주파수 성분에 더 높은 가중치를 부여하여
    손실을 계산한다.

    Args:
        alpha (float): 동적 가중치의 지수. 클수록 hard 주파수를 더 강조.
                       원 논문 권장값: 1.0
        patch_factor (int): 패치를 더 작게 분할할 배율. 1이면 분할 없음.
        ave_spectrum (bool): True이면 배치 평균 스펙트럼으로 손실 계산.
        log_matrix (bool): True이면 주파수 진폭에 로그 스케일 적용.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        freq_mask_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.freq_mask_ratio = freq_mask_ratio

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """2D FFT 변환 후 실수/허수 파트를 마지막 차원으로 stack.

        Args:
            x: (B, H, W) 형태의 패치 텐서.

        Returns:
            (B, H, W, 2) 형태. 마지막 차원 [0]=real, [1]=imag.
        """
        # ortho 정규화: 순전파/역전파 모두 에너지 보존
        freq = torch.fft.fft2(x, norm="ortho")
        # fftshift: DC(저주파) 성분을 중앙으로 이동
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        return torch.stack([freq.real, freq.imag], dim=-1)

    def _compute_loss(
        self,
        pred_freq: torch.Tensor,
        gt_freq: torch.Tensor,
    ) -> torch.Tensor:
        """동적 가중치를 이용한 주파수 공간 손실 계산.

        Args:
            pred_freq: (B, H, W, 2) 예측 주파수 맵.
            gt_freq:   (B, H, W, 2) 정답 주파수 맵.

        Returns:
            scalar 손실값.
        """
        # 주파수별 제곱 오차: (B, H, W)
        sq_diff = (pred_freq - gt_freq).pow(2).sum(dim=-1)

        # ===== [마스킹 로직 추가] =====
        if self.freq_mask_ratio < 1.0:
            B, H, W = sq_diff.shape
            cy, cx = H // 2, W // 2
            
            # 중심 기준 픽셀 좌표 거리 계산
            y = torch.arange(H, device=sq_diff.device).view(-1, 1) - cy
            x = torch.arange(W, device=sq_diff.device).view(1, -1) - cx
            r = torch.sqrt(y**2 + x**2)
            
            # 최대 반경 계산
            max_r = torch.sqrt(torch.tensor(cy**2 + cx**2, dtype=torch.float32, device=sq_diff.device))
            
            # 허용할 반경
            mask_radius = max_r * self.freq_mask_ratio
            
            # 마스크 생성 (반경 내부=1, 외부=0)
            mask = (r <= mask_radius).float().view(1, H, W)
            
            # 오차에 마스크 적용 (초고주파 노이즈 오차 무시)
            sq_diff = sq_diff * mask
        # ===== [마스킹 로직 추가 끝] =====

        # 동적 가중치: hard 주파수(오차가 큰 곳)에 높은 가중치 부여
        # detach() 로 가중치 자체는 역전파에서 제외
        weight = sq_diff.detach().pow(self.alpha / 2.0)
        # 배치 평균으로 정규화하여 스케일 안정화
        weight = weight / (weight.mean() + 1e-8)

        return (weight * sq_diff).mean()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor = None,
        prior_template: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt is not None and gt.dim() == 4:
            gt = gt.squeeze(1)

        B, H, W = pred.shape

        if prior_template is not None:
            # --- [Path A: Dataset-level Prior 방식] ---
            # 1. 패치별 정규화 (추출 스크립트와 동일한 조건: 평균 0, 분산 1)
            pred_flat = pred.view(B, -1)
            pred_mean = pred_flat.mean(dim=1, keepdim=True).view(B, 1, 1)
            pred_std = pred_flat.std(dim=1, keepdim=True).view(B, 1, 1) + 1e-8
            pred_norm = (pred - pred_mean) / pred_std

            # 2. 2D FFT 변환 및 진폭(Amplitude) 추출
            pred_freq = self.tensor2freq(pred_norm)  # (B, H, W, 2)
            pred_amp = torch.sqrt(pred_freq[..., 0]**2 + pred_freq[..., 1]**2 + 1e-8)

            # 3. 템플릿과 비교 (Squared Difference)
            sq_diff = (pred_amp - prior_template.to(pred_amp.device)).pow(2)
        else:
            # --- [기존 방식: GT(정답지) 기반 비교] ---
            pred_freq = self.tensor2freq(pred)
            gt_freq = self.tensor2freq(gt)
            sq_diff = (pred_freq - gt_freq).pow(2).sum(dim=-1)

        # ===== [마스킹 로직] =====
        if self.freq_mask_ratio < 1.0:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=sq_diff.device).view(-1, 1) - cy
            x = torch.arange(W, device=sq_diff.device).view(1, -1) - cx
            r = torch.sqrt(y**2 + x**2)
            max_r = torch.sqrt(torch.tensor(cy**2 + cx**2, dtype=torch.float32, device=sq_diff.device))
            mask_radius = max_r * self.freq_mask_ratio
            mask = (r <= mask_radius).float().view(1, H, W)
            sq_diff = sq_diff * mask

        # 동적 가중치 (Hard frequency 강조)
        weight = sq_diff.detach().pow(self.alpha / 2.0)
        weight = weight / (weight.mean() + 1e-8)

        return (weight * sq_diff).mean()
