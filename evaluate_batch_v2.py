"""
evaluate_batch_v2.py
====================
evaluate_batch.py 를 기반으로 VIF, FSIM, HFEN 지표를 추가한 확장 버전.

추가된 지표:
  VIF  (Visual Information Fidelity)  : 높을수록 좋음 [0, 1+]
  FSIM (Feature Similarity Index)     : 높을수록 좋음 [0, 1]
  HFEN (High Frequency Error Norm)    : 낮을수록 좋음 [0, inf)

새로 추가된 동작 플래그:
  VIF_3AXES  : True  → axial / coronal / sagittal 3축 평균 VIF
               False → axial only
  FSIM_3AXES : True  → axial / coronal / sagittal 3축 평균 FSIM
               False → axial only
  HFEN_SIGMA : LoG 필터의 가우시안 sigma (권장: 1.0~2.5)
               - 낮은 값: 더 세밀한 고주파 구조 강조
               - 높은 값: 더 넓은 범위의 엣지/구조 강조

의존성:
  기존: nibabel, numpy, torch, lpips, scikit-image, pandas
  신규: torchmetrics (VIF), piq (FSIM), scipy (HFEN)

  설치 명령:
    pip install torchmetrics piq scipy
"""

import os, re, glob
import nibabel as nib
import numpy as np
import torch
import lpips
import pandas as pd
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity as ssim
from typing import Optional

# torchmetrics VIF
from torchmetrics.functional.image import visual_information_fidelity

# piq FSIM — 없으면 경고만 출력하고 계속 실행
try:
    import piq
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False
    print("[WARN] piq 라이브러리가 없습니다. FSIM은 None으로 기록됩니다.")
    print("       설치: pip install piq")


# ==============================================================================
# ★ 설정부 — 실행 전 이 섹션만 수정하세요
# ==============================================================================

# ── 기존 동작 플래그 ───────────────────────────────────────────────────────────
USE_SSIM_BBOX_CROP = True   # True: bbox 크롭 후 SSIM  |  False: 전체 슬라이스
LPIPS_3AXES        = True   # True: 3축 평균 LPIPS      |  False: axial only

# ── 신규 동작 플래그 ───────────────────────────────────────────────────────────
VIF_3AXES  = True    # True: 3축 평균 VIF   |  False: axial only
FSIM_3AXES = True    # True: 3축 평균 FSIM  |  False: axial only
HFEN_SIGMA = 1.5     # LoG 필터 sigma (float). MRI 등방성 재구성 권장값: 1.0~2.5

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
TEST_CSV   = "/dshome/ddualab/dongnyeok/NeSVoR/test.csv"
RESULT_CSV = "/dshome/ddualab/dongnyeok/eval_results/ablation_results_v2.csv"

GT_ROOT = "/data/BraTS2020_TrainingData"

METHODS = {
    "Ours"      : "/dshome/ddualab/dongnyeok/NeSVoR/recon_IC",
    "Trilinear" : "/dshome/ddualab/dongnyeok/densenet121/trilinear",
    "Bspline"   : "/dshome/ddualab/dongnyeok/densenet121/bspline_cubic",
    "NeSVoR"    : "/dshome/ddualab/dongnyeok/NeSVoR_original/recon_base",
    "CuNeRF"    : "/dshome/ddualab/dongnyeok/Cunerf_ori/recon",
    "ArSSR"     : "/dshome/ddualab/dongnyeok/arssr/output",
    "ECLAIR"    : "/dshome/ddualab/dongnyeok/eclare/recon",
    "no_ff"     : "/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation/no_ff",
    "no_gating" : "/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation/no_gating",
    "no_hm"     : "/dshome/ddualab/dongnyeok/NeSVoR/recon_ablation/no_hm",
}

MODALITIES = ["flair", "t1ce"]

# SSIM/LPIPS/VIF/FSIM 계산 시 유효 슬라이스로 인정할 최소 brain 복셀 수
MIN_SLICE_VOXELS = 110


# ==============================================================================
# 파일 탐색 (기존과 동일)
# ==============================================================================

def find_modality_file(dataset_root: str, patient_id: str, modality: str) -> Optional[str]:
    """
    환자 폴더 안에서 modality 키워드를 포함하는 NIfTI 파일을 탐색합니다.

    지원 파일명 예시:
      - BraTS20_Training_070_flair.nii          (BraTS 원본 규칙)
      - 088_t1ce_4x5_E1.nii.gz                  (short ID 규칙)
    """
    patient_dir = os.path.join(dataset_root, patient_id)
    if not os.path.isdir(patient_dir):
        return None

    m = re.search(r'(\d+)$', patient_id)
    short_id = m.group(1) if m else ""

    patterns = [
        os.path.join(patient_dir, f"*{patient_id}*{modality}*.nii*"),
        os.path.join(patient_dir, f"*{short_id}*{modality}*.nii*"),
        os.path.join(patient_dir, f"*{modality}*{short_id}*.nii*"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


# ==============================================================================
# 공간 정렬 (기존과 동일)
# ==============================================================================

def robust_align_recon(orig_nib, recon_nib):
    """Orientation 정렬 → Shape 불일치 시 Affine 오프셋 기반 패딩"""
    orig_shape = orig_nib.shape[:3]
    orig_ornt  = nib.io_orientation(orig_nib.affine)
    recon_ornt = nib.io_orientation(recon_nib.affine)

    if not np.array_equal(orig_ornt, recon_ornt):
        transform = nib.ornt_transform(recon_ornt, orig_ornt)
        recon_nib = recon_nib.as_reoriented(transform)

    recon_shape = recon_nib.shape[:3]
    recon_data  = recon_nib.get_fdata().astype(np.float32)

    if orig_shape != recon_shape:
        inv_orig_affine       = np.linalg.inv(orig_nib.affine)
        recon_origin_physical = recon_nib.affine @ np.array([0, 0, 0, 1])
        offset_voxels         = inv_orig_affine @ recon_origin_physical
        x_off, y_off, z_off   = np.round(offset_voxels[:3]).astype(int)

        rx, ry, rz = recon_shape
        aligned_data = np.zeros(orig_shape, dtype=np.float32)

        tgt_x0, tgt_x1 = max(0, x_off),  min(orig_shape[0], x_off + rx)
        tgt_y0, tgt_y1 = max(0, y_off),  min(orig_shape[1], y_off + ry)
        tgt_z0, tgt_z1 = max(0, z_off),  min(orig_shape[2], z_off + rz)

        src_x0, src_x1 = max(0, -x_off), max(0, -x_off) + (tgt_x1 - tgt_x0)
        src_y0, src_y1 = max(0, -y_off), max(0, -y_off) + (tgt_y1 - tgt_y0)
        src_z0, src_z1 = max(0, -z_off), max(0, -z_off) + (tgt_z1 - tgt_z0)

        if (tgt_x1 > tgt_x0) and (tgt_y1 > tgt_y0) and (tgt_z1 > tgt_z0):
            aligned_data[tgt_x0:tgt_x1, tgt_y0:tgt_y1, tgt_z0:tgt_z1] = \
                recon_data[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
        recon_data = aligned_data

    new_header = orig_nib.header.copy()
    new_header.set_data_shape(recon_data.shape)
    new_header['scl_slope'] = 1.0
    new_header['scl_inter'] = 0.0
    return nib.Nifti1Image(recon_data, orig_nib.affine, new_header)


# ==============================================================================
# 스케일 매칭 & 정규화 (기존과 동일)
# ==============================================================================

def normalize_pair(orig_array: np.ndarray, recon_array: np.ndarray,
                   brain_mask: np.ndarray):
    """
    1) 원본 스케일에서 최소제곱 k 매칭
    2) GT 기준 0.1~99.9 percentile로 [0, 1] 공유 정규화
    Returns: orig_norm, recon_norm, k_val
    """
    k = (np.sum(orig_array[brain_mask] * recon_array[brain_mask]) /
         (np.sum(recon_array[brain_mask] ** 2) + 1e-8))
    recon_matched = recon_array * k

    mn = np.percentile(orig_array[brain_mask], 0.1)
    mx = np.percentile(orig_array[brain_mask], 99.9)

    orig_norm  = np.clip((orig_array    - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    recon_norm = np.clip((recon_matched - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    orig_norm[~brain_mask]  = 0.0
    recon_norm[~brain_mask] = 0.0

    return orig_norm, recon_norm, float(k)


# ==============================================================================
# 기존 지표 함수 (변경 없음)
# ==============================================================================

def calc_psnr(ref: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    mse = np.mean((ref[mask] - recon[mask]) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))

def calc_snr(ref: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    sig   = np.sum(ref[mask] ** 2)
    noise = np.sum((ref[mask] - recon[mask]) ** 2)
    return float('inf') if noise == 0 else float(10 * np.log10(sig / noise))

def calc_nrmse(ref: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((ref[mask] - recon[mask]) ** 2))
    norm = np.sqrt(np.mean(ref[mask] ** 2))
    return float(rmse / (norm + 1e-8))

def calc_ssim_3axes(ref_vol: np.ndarray, recon_vol: np.ndarray,
                    mask_vol: np.ndarray, data_range: float = 1.0) -> tuple:
    """
    3축(axial/coronal/sagittal) SSIM.
    USE_SSIM_BBOX_CROP=True  : 각 슬라이스의 brain bounding box만 크롭 후 계산
    USE_SSIM_BBOX_CROP=False : 전체 슬라이스 대상 계산
    """
    axes = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    per_axis = {}

    for axis_name, axis in axes.items():
        scores = []
        for i in range(ref_vol.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i

            ref_s   = ref_vol[tuple(idx)]
            recon_s = recon_vol[tuple(idx)]
            mask_s  = mask_vol[tuple(idx)]

            if np.sum(mask_s) <= MIN_SLICE_VOXELS:
                continue

            if USE_SSIM_BBOX_CROP:
                rows = np.any(mask_s, axis=1)
                cols = np.any(mask_s, axis=0)
                if not rows.any() or not cols.any():
                    continue
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                if (r1 - r0 + 1) < 7 or (c1 - c0 + 1) < 7:
                    continue
                ref_s   = ref_s[r0:r1+1, c0:c1+1]
                recon_s = recon_s[r0:r1+1, c0:c1+1]
            else:
                if ref_s.shape[0] < 7 or ref_s.shape[1] < 7:
                    continue

            scores.append(ssim(ref_s, recon_s, data_range=data_range, win_size=7))

        per_axis[axis_name] = float(np.mean(scores)) if scores else 0.0

    mean_ssim = float(np.mean(list(per_axis.values())))
    return mean_ssim, per_axis


def calc_lpips_3d(ref_vol: np.ndarray, recon_vol: np.ndarray,
                  mask_vol: np.ndarray, loss_fn, device) -> tuple:
    """
    LPIPS_3AXES=True  : axial / coronal / sagittal 3축 평균
    LPIPS_3AXES=False : axial only
    """
    def _score_along_axis(axis: int) -> float:
        scores = []
        for i in range(ref_vol.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i
            mask_s = mask_vol[tuple(idx)]
            if np.sum(mask_s) <= MIN_SLICE_VOXELS:
                continue
            r_s  = ref_vol[tuple(idx)]
            rc_s = recon_vol[tuple(idx)]
            r_t  = (torch.from_numpy(r_s  * 2.0 - 1.0).float()
                    .unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device))
            rc_t = (torch.from_numpy(rc_s * 2.0 - 1.0).float()
                    .unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device))
            with torch.no_grad():
                scores.append(loss_fn(r_t, rc_t).item())
        return float(np.mean(scores)) if scores else 0.0

    axial_score = _score_along_axis(2)

    if LPIPS_3AXES:
        coronal_score  = _score_along_axis(1)
        sagittal_score = _score_along_axis(0)
        mean_lpips = float(np.mean([axial_score, coronal_score, sagittal_score]))
        axes_dict  = {'axial': axial_score, 'coronal': coronal_score, 'sagittal': sagittal_score}
    else:
        mean_lpips = axial_score
        axes_dict  = {'axial': axial_score, 'coronal': None, 'sagittal': None}

    return mean_lpips, axes_dict


# ==============================================================================
# 신규 지표 함수
# ==============================================================================

def calc_vif_3d(ref_vol: np.ndarray, recon_vol: np.ndarray,
                mask_vol: np.ndarray, device) -> tuple:
    """
    VIF (Visual Information Fidelity) — 슬라이스별 계산 후 3축 평균.

    torchmetrics.functional.image.visual_information_fidelity 사용.
      - 입력: (1, 1, H, W) float32 텐서, [0, 1] 범위
      - H, W 모두 41 이상인 슬라이스만 유효 (torchmetrics 내부 제한)
      - VIF_3AXES=True: axial/coronal/sagittal 3축 평균
      - VIF_3AXES=False: axial only

    Returns:
        (mean_vif, axes_dict)
        axes_dict keys: 'axial', 'coronal', 'sagittal'
    """
    MIN_VIF_SIZE = 41  # torchmetrics VIF 요구 최소 H/W

    def _score_along_axis(axis: int) -> float:
        scores = []
        for i in range(ref_vol.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i
            mask_s = mask_vol[tuple(idx)]

            if np.sum(mask_s) <= MIN_SLICE_VOXELS:
                continue

            r_s  = ref_vol[tuple(idx)]
            rc_s = recon_vol[tuple(idx)]

            # H/W 크기 검증 (torchmetrics VIF 내부 요구사항)
            if r_s.shape[0] < MIN_VIF_SIZE or r_s.shape[1] < MIN_VIF_SIZE:
                continue

            # (1, 1, H, W) 텐서 — [0, 1] 범위 그대로 사용
            r_t  = torch.from_numpy(r_s ).float().unsqueeze(0).unsqueeze(0).to(device)
            rc_t = torch.from_numpy(rc_s).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                score = visual_information_fidelity(rc_t, r_t, reduction='mean')
                scores.append(score.item())

        return float(np.mean(scores)) if scores else 0.0

    axial_score = _score_along_axis(2)

    if VIF_3AXES:
        coronal_score  = _score_along_axis(1)
        sagittal_score = _score_along_axis(0)
        mean_vif  = float(np.mean([axial_score, coronal_score, sagittal_score]))
        axes_dict = {'axial': axial_score, 'coronal': coronal_score, 'sagittal': sagittal_score}
    else:
        mean_vif  = axial_score
        axes_dict = {'axial': axial_score, 'coronal': None, 'sagittal': None}

    return mean_vif, axes_dict


def calc_fsim_3d(ref_vol: np.ndarray, recon_vol: np.ndarray,
                 mask_vol: np.ndarray, device) -> tuple:
    """
    FSIM (Feature Similarity Index) — 슬라이스별 계산 후 3축 평균.

    piq.fsim 사용.
      - 입력: (1, 1, H, W) float32 텐서, [0, 1] 범위
      - FSIM_3AXES=True: axial/coronal/sagittal 3축 평균
      - FSIM_3AXES=False: axial only
      - piq 미설치 시 (None, axes_dict with None) 반환

    Returns:
        (mean_fsim, axes_dict)
        axes_dict keys: 'axial', 'coronal', 'sagittal'
    """
    if not PIQ_AVAILABLE:
        return None, {'axial': None, 'coronal': None, 'sagittal': None}

    def _score_along_axis(axis: int) -> float:
        scores = []
        for i in range(ref_vol.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i
            mask_s = mask_vol[tuple(idx)]

            if np.sum(mask_s) <= MIN_SLICE_VOXELS:
                continue

            r_s  = ref_vol[tuple(idx)]
            rc_s = recon_vol[tuple(idx)]

            if r_s.shape[0] < 7 or r_s.shape[1] < 7:
                continue

            # piq.fsim: (N, C, H, W), [0, 1] float32
            r_t  = torch.from_numpy(r_s ).float().unsqueeze(0).unsqueeze(0).to(device)
            rc_t = torch.from_numpy(rc_s).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                score = piq.fsim(rc_t, r_t, data_range=1.0, chromatic=False, reduction='mean')
                scores.append(score.item())

        return float(np.mean(scores)) if scores else 0.0

    axial_score = _score_along_axis(2)

    if FSIM_3AXES:
        coronal_score  = _score_along_axis(1)
        sagittal_score = _score_along_axis(0)
        mean_fsim = float(np.mean([axial_score, coronal_score, sagittal_score]))
        axes_dict = {'axial': axial_score, 'coronal': coronal_score, 'sagittal': sagittal_score}
    else:
        mean_fsim = axial_score
        axes_dict = {'axial': axial_score, 'coronal': None, 'sagittal': None}

    return mean_fsim, axes_dict


def calc_hfen_3d(ref_vol: np.ndarray, recon_vol: np.ndarray,
                 mask: np.ndarray, sigma: float = 1.5) -> float:
    """
    HFEN (High Frequency Error Norm) — 3D 볼륨 단위 계산.

    공식:
        HFEN = ||LoG(recon)[mask] - LoG(ref)[mask]||_2
               / (||LoG(ref)[mask]||_2 + eps)

    scipy.ndimage.gaussian_laplace 로 LoG 필터 구현.
    3D 볼륨 전체에 적용하므로 through-plane 방향 고주파 구조도 함께 평가.

    Args:
        ref_vol:   GT 볼륨 ([0,1] 정규화)
        recon_vol: 재구성 볼륨 ([0,1] 정규화)
        mask:      Brain mask (bool array)
        sigma:     LoG 필터의 가우시안 sigma. 기본값 1.5.
                   낮을수록 세밀한 고주파, 높을수록 넓은 구조 범위를 강조.

    Returns:
        HFEN 값 (float). 낮을수록 고주파 구조가 잘 보존됨.
    """
    log_ref   = gaussian_laplace(ref_vol,   sigma=sigma)
    log_recon = gaussian_laplace(recon_vol, sigma=sigma)

    diff_norm = np.sqrt(np.sum((log_recon[mask] - log_ref[mask]) ** 2))
    ref_norm  = np.sqrt(np.sum(log_ref[mask] ** 2))

    return float(diff_norm / (ref_norm + 1e-8))


# ==============================================================================
# 단일 파일 쌍 평가 (확장)
# ==============================================================================

def evaluate_pair(gt_path: str, recon_path: str, lpips_fn, device) -> dict:
    orig_nib  = nib.load(gt_path)
    recon_nib = nib.load(recon_path)

    recon_aligned_nib = robust_align_recon(orig_nib, recon_nib)

    orig_array  = orig_nib.get_fdata().astype(np.float32)
    recon_array = recon_aligned_nib.get_fdata().astype(np.float32)
    recon_array = np.nan_to_num(recon_array, nan=0.0, posinf=0.0, neginf=0.0)

    assert orig_array.shape == recon_array.shape, \
        f"Shape mismatch after alignment: {orig_array.shape} vs {recon_array.shape}"

    brain_mask = orig_array > 0

    orig_norm, recon_norm, k_val = normalize_pair(orig_array, recon_array, brain_mask)

    # ── 기존 지표 ──────────────────────────────────────────────────────────────
    psnr_val  = calc_psnr(orig_norm, recon_norm, brain_mask)
    snr_val   = calc_snr(orig_norm, recon_norm, brain_mask)
    nrmse_val = calc_nrmse(orig_norm, recon_norm, brain_mask)

    ssim_val,  ssim_axes  = calc_ssim_3axes(orig_norm, recon_norm, brain_mask)
    lpips_val, lpips_axes = calc_lpips_3d(orig_norm, recon_norm, brain_mask, lpips_fn, device)

    # ── 신규 지표 ──────────────────────────────────────────────────────────────
    vif_val,  vif_axes  = calc_vif_3d(orig_norm, recon_norm, brain_mask, device)
    fsim_val, fsim_axes = calc_fsim_3d(orig_norm, recon_norm, brain_mask, device)
    hfen_val            = calc_hfen_3d(orig_norm, recon_norm, brain_mask, sigma=HFEN_SIGMA)

    return {
        # 기존
        "PSNR"            : psnr_val,
        "SNR"             : snr_val,
        "NRMSE"           : nrmse_val,
        "SSIM"            : ssim_val,
        "SSIM_axial"      : ssim_axes['axial'],
        "SSIM_coronal"    : ssim_axes['coronal'],
        "SSIM_sagittal"   : ssim_axes['sagittal'],
        "LPIPS"           : lpips_val,
        "LPIPS_axial"     : lpips_axes['axial'],
        "LPIPS_coronal"   : lpips_axes['coronal'],
        "LPIPS_sagittal"  : lpips_axes['sagittal'],
        # 신규
        "VIF"             : vif_val,
        "VIF_axial"       : vif_axes['axial'],
        "VIF_coronal"     : vif_axes['coronal'],
        "VIF_sagittal"    : vif_axes['sagittal'],
        "FSIM"            : fsim_val,
        "FSIM_axial"      : fsim_axes['axial'],
        "FSIM_coronal"    : fsim_axes['coronal'],
        "FSIM_sagittal"   : fsim_axes['sagittal'],
        "HFEN"            : hfen_val,
        # 메타
        "k_scale"         : k_val,
    }


# ==============================================================================
# 메인 배치 루프
# ==============================================================================

def main():
    # 1. test.csv에서 환자 ID 로드
    df_test     = pd.read_csv(TEST_CSV)
    patient_ids = df_test['Brats20ID'].str.strip().tolist()
    print(f"[INFO] 평가 대상 환자 수: {len(patient_ids)}")

    # 2. 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 디바이스: {device}")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # 3. 설정 출력
    print(f"[CONFIG] USE_SSIM_BBOX_CROP = {USE_SSIM_BBOX_CROP}")
    print(f"[CONFIG] LPIPS_3AXES        = {LPIPS_3AXES}")
    print(f"[CONFIG] VIF_3AXES          = {VIF_3AXES}")
    print(f"[CONFIG] FSIM_3AXES         = {FSIM_3AXES}")
    print(f"[CONFIG] HFEN_SIGMA         = {HFEN_SIGMA}")
    print(f"[CONFIG] PIQ (FSIM)         = {'available' if PIQ_AVAILABLE else 'NOT AVAILABLE — FSIM will be None'}")
    print("=" * 70)

    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

    results = []
    total   = len(patient_ids) * len(MODALITIES) * len(METHODS)
    done    = 0

    for patient_id in patient_ids:
        for modality in MODALITIES:
            gt_path = find_modality_file(GT_ROOT, patient_id, modality)
            if gt_path is None:
                print(f"[WARN] GT 파일 없음: {patient_id} / {modality}")
                continue

            for method_name, method_root in METHODS.items():
                done += 1
                recon_path = find_modality_file(method_root, patient_id, modality)

                base_row = {
                    "patient_id" : patient_id,
                    "modality"   : modality,
                    "method"     : method_name,
                }
                null_metrics = {k: None for k in [
                    "PSNR", "SNR", "NRMSE",
                    "SSIM",  "SSIM_axial",  "SSIM_coronal",  "SSIM_sagittal",
                    "LPIPS", "LPIPS_axial", "LPIPS_coronal", "LPIPS_sagittal",
                    "VIF",   "VIF_axial",   "VIF_coronal",   "VIF_sagittal",
                    "FSIM",  "FSIM_axial",  "FSIM_coronal",  "FSIM_sagittal",
                    "HFEN",
                    "k_scale",
                ]}

                if recon_path is None:
                    print(f"[{done:>3}/{total}] [SKIP] {method_name:<12} {patient_id} {modality} — 파일 없음")
                    results.append({**base_row, **null_metrics})
                    continue

                print(f"[{done:>3}/{total}] {method_name:<12} | {patient_id} | {modality}")
                try:
                    metrics = evaluate_pair(gt_path, recon_path, lpips_fn, device)
                    results.append({**base_row, **metrics})
                    fsim_str = f"{metrics['FSIM']:.4f}" if metrics['FSIM'] is not None else "N/A"
                    print(
                        f"         PSNR={metrics['PSNR']:7.3f} dB  "
                        f"SSIM={metrics['SSIM']:.4f}  "
                        f"NRMSE={metrics['NRMSE']:.4f}  "
                        f"LPIPS={metrics['LPIPS']:.4f}  "
                        f"VIF={metrics['VIF']:.4f}  "
                        f"FSIM={fsim_str}  "
                        f"HFEN={metrics['HFEN']:.4f}"
                    )
                except Exception as e:
                    print(f"         [ERROR] {e}")
                    results.append({**base_row, **null_metrics})

    # 4. 결과 저장
    df_result = pd.DataFrame(results)
    df_result.to_csv(RESULT_CSV, index=False)
    print(f"\n[DONE] 전체 결과 저장 완료: {RESULT_CSV}")

    # 5. 집계 출력 (터미널 확인용)
    metric_cols = [
        "PSNR",
        "SSIM",  "SSIM_axial",  "SSIM_coronal",  "SSIM_sagittal",
        "NRMSE",
        "LPIPS",
        "VIF",   "VIF_axial",   "VIF_coronal",   "VIF_sagittal",
        "FSIM",  "FSIM_axial",  "FSIM_coronal",  "FSIM_sagittal",
        "HFEN",
    ]
    print("\n" + "=" * 70)
    print("  집계 결과 (mean +/- std)")
    print("=" * 70)
    for modality in MODALITIES:
        print(f"\n── {modality.upper()} ──")
        sub     = df_result[df_result['modality'] == modality].copy()
        summary = sub.groupby('method')[metric_cols].agg(['mean', 'std'])
        print(summary.to_string())


if __name__ == "__main__":
    main()
