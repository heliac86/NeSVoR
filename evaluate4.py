import nibabel as nib
import nibabel.processing
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------
# 1. 파일 경로 설정
# ---------------------------------------------------------
ORIG_PATH  = "/data/BraTS2020_TrainingData/BraTS20_Training_003/BraTS20_Training_003_flair.nii"
# RECON_PATH = "003_flair_06_multiview.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_06.nii.gz"

# RECON_PATH = "003_flair_baseline.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_baseline.nii.gz"

# RECON_PATH = "recon_BraTS20_Training_003_flair.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_gate.nii.gz"

# RECON_PATH = "/dshome/ddualab/dongnyeok/Cunerf_0213/save/freq_08/BraTS20_Training_003_flair/eval/ours.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_cunerf_ex.nii.gz"

RECON_PATH = "003_flair_10.nii.gz"
ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_10.nii.gz"


# ---------------------------------------------------------
# 2. 정렬 함수
# ---------------------------------------------------------
def robust_align_recon(orig_nib, recon_nib, save_path=None):
    orig_axcodes  = nib.aff2axcodes(orig_nib.affine)
    recon_axcodes = nib.aff2axcodes(recon_nib.affine)
    orig_shape    = orig_nib.shape[:3]
    recon_shape   = recon_nib.shape[:3]

    print(f"[DEBUG] 원본   : orientation={orig_axcodes}, shape={orig_shape}")
    print(f"[DEBUG] 재구성 : orientation={recon_axcodes}, shape={recon_shape}")

    if orig_shape == recon_shape:
        if orig_axcodes == recon_axcodes:
            print("[INFO] Case A: Orientation 일치 → affine 직접 복사")
            recon_data = recon_nib.get_fdata().astype(np.float32)
        else:
            print(f"[INFO] Case B: Orientation 불일치 ({recon_axcodes}→{orig_axcodes}) → 배열 재배치 후 affine 복사")
            orig_ornt  = nib.io_orientation(orig_nib.affine)
            recon_ornt = nib.io_orientation(recon_nib.affine)
            transform  = nib.ornt_transform(recon_ornt, orig_ornt)
            recon_reoriented = recon_nib.as_reoriented(transform)
            recon_data = recon_reoriented.get_fdata().astype(np.float32)

            if recon_data.shape[:3] != orig_shape:
                raise ValueError(
                    f"[ERROR] Reorientation 후 shape 불일치: {recon_data.shape[:3]} vs {orig_shape}.\n"
                    "Case C(resample) 방식으로 처리하거나 파일을 점검하세요."
                )

        new_header = orig_nib.header.copy()
        new_header.set_data_shape(recon_data.shape)
        new_header['scl_slope'] = 1.0
        new_header['scl_inter'] = 0.0
        recon_aligned = nib.Nifti1Image(recon_data, orig_nib.affine, new_header)
        orig_final = orig_nib
    else:
        print(f"[INFO] Case C: Shape 불일치 {recon_shape} vs {orig_shape} → canonical resample")
        orig_canonical  = nib.as_closest_canonical(orig_nib)
        recon_canonical = nib.as_closest_canonical(recon_nib)
        recon_aligned = nibabel.processing.resample_from_to(
            recon_canonical, orig_canonical, order=3
        )
        orig_final = orig_canonical

    if save_path:
        nib.save(recon_aligned, save_path)
        print(f"[INFO] 정렬 파일 저장: '{save_path}'")

    return orig_final, recon_aligned

# ---------------------------------------------------------
# 3. 평가 지표 함수
# ---------------------------------------------------------
def calculate_psnr_masked(ref, recon, mask):
    mse = np.mean((ref[mask] - recon[mask]) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def calculate_snr(ref, recon, mask):
    signal_power = np.sum(ref[mask] ** 2)
    noise_power  = np.sum((ref[mask] - recon[mask]) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_nrmse(ref, recon, mask):
    rmse        = np.sqrt(np.mean((ref[mask] - recon[mask]) ** 2))
    norm_factor = np.sqrt(np.mean(ref[mask] ** 2))
    return float(rmse / (norm_factor + 1e-8))

def calculate_ssim_3axes(ref_vol, recon_vol, mask_vol, data_range=1.0, min_slice_voxels=100):
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
            if np.sum(mask_s) > min_slice_voxels:
                scores.append(ssim(ref_s, recon_s, data_range=data_range))
        per_axis[axis_name] = float(np.mean(scores)) if scores else 0.0
    mean_ssim = float(np.mean(list(per_axis.values())))
    return mean_ssim, per_axis

def calculate_lpips_3d(ref_vol, recon_vol, mask_vol, loss_fn, device, min_slice_voxels=100):
    scores = []
    for i in range(ref_vol.shape[2]):
        if np.sum(mask_vol[:, :, i]) > min_slice_voxels:
            r  = torch.from_numpy(ref_vol[:, :, i]  * 2.0 - 1.0).float()
            rc = torch.from_numpy(recon_vol[:, :, i] * 2.0 - 1.0).float()
            r  = r.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            rc = rc.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            with torch.no_grad():
                scores.append(loss_fn(r, rc).item())
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------
# 4. 메인
# ---------------------------------------------------------
print("=" * 55)
print("  데이터 로드 중...")
print("=" * 55)

orig_nib  = nib.load(ORIG_PATH)
recon_nib = nib.load(RECON_PATH)

print(f"\n[헤더] 원본   sform_code={orig_nib.header['sform_code']}, qform_code={orig_nib.header['qform_code']}")
print(f"[헤더] 재구성 sform_code={recon_nib.header['sform_code']}, qform_code={recon_nib.header['qform_code']}")

print("\n정렬 중...")
orig_final, recon_aligned = robust_align_recon(orig_nib, recon_nib, save_path=ALIGNED_RECON_SAVE_PATH)

orig_array  = orig_final.get_fdata().astype(np.float32)
recon_array = recon_aligned.get_fdata().astype(np.float32)

nan_count = np.sum(~np.isfinite(recon_array))
if nan_count > 0:
    recon_array = np.nan_to_num(recon_array, nan=0.0, posinf=0.0, neginf=0.0)

assert orig_array.shape == recon_array.shape, "[ERROR] Shape 불일치! 정렬 실패."

brain_mask = orig_array > 0
print(f"brain_mask — GT 기준 생성, 유효 voxel 수: {np.sum(brain_mask):,}")

# ─────────────────────────────────────────────────
# ★ 수정 1: [0, 1]로 자르기(Clip) 전 원본 스케일에서 강도 보정 (k 매칭)
# ─────────────────────────────────────────────────
print("\n원본 스케일 기반 강도 보정(k 매칭) 수행 중...")
print(f" - 보정 전 GT 평균: {orig_array[brain_mask].mean():.2f}")
print(f" - 보정 전 Recon 평균: {recon_array[brain_mask].mean():.2f}")

k_raw = (np.sum(orig_array[brain_mask] * recon_array[brain_mask]) /
         (np.sum(recon_array[brain_mask] ** 2) + 1e-8))

print(f" - 추정 스케일 k_raw = {k_raw:.4f}")
recon_matched = recon_array * k_raw

print(f" - 보정 후 Recon 평균: {recon_matched[brain_mask].mean():.2f}")

# ─────────────────────────────────────────────────
# ★ 수정 2: 보정된 볼륨을 GT 기준으로 [0, 1] 공유 정규화
# ─────────────────────────────────────────────────
print("\n정규화 중 (GT 기준 0~1 공유 정규화)...")
mn = np.percentile(orig_array[brain_mask], 0.1)
mx = np.percentile(orig_array[brain_mask], 99.9)
print(f"  정규화 하한(0.1%ile): {mn:.2f}, 상한(99.9%ile): {mx:.2f}")

orig_norm = np.clip((orig_array - mn) / (mx - mn + 1e-8), 0.0, 1.0)
orig_norm[~brain_mask] = 0.0

# 이미 k_raw로 보정된 recon_matched를 정규화
recon_norm = np.clip((recon_matched - mn) / (mx - mn + 1e-8), 0.0, 1.0)
recon_norm[~brain_mask] = 0.0

# ---------------------------------------------------------
# LPIPS 초기화 및 오차 확인
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nLPIPS 모델 로드 중 (AlexNet backbone)...")
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

diff_map = orig_norm - recon_norm
diff_map[~brain_mask] = 0

print("\n[구역별 오차]")
h, w, d = orig_norm.shape
for zi, zname in enumerate(['inf', 'sup']):
    for yi, yname in enumerate(['ant', 'pos']):
        for xi, xname in enumerate(['L', 'R']):
            mask_r = brain_mask[
                xi*h//2:(xi+1)*h//2,
                yi*w//2:(yi+1)*w//2,
                zi*d//2:(zi+1)*d//2
            ]
            region = diff_map[
                xi*h//2:(xi+1)*h//2,
                yi*w//2:(yi+1)*w//2,
                zi*d//2:(zi+1)*d//2
            ]
            if mask_r.sum() > 0:
                print(f"{xname}-{yname}-{zname}: mean_diff={region[mask_r].mean():.4f}")

# ---------------------------------------------------------
# 최종 지표 계산
# ---------------------------------------------------------
print("\n지표 계산 중...")
# 이제 recon_norm 자체에 보정이 반영되어 있으므로 그대로 사용합니다.
current_psnr              = calculate_psnr_masked(orig_norm, recon_norm, brain_mask)
current_snr               = calculate_snr(orig_norm, recon_norm, brain_mask)
current_nrmse             = calculate_nrmse(orig_norm, recon_norm, brain_mask)
current_ssim, ssim_per_ax = calculate_ssim_3axes(orig_norm, recon_norm, brain_mask)
current_lpips             = calculate_lpips_3d(orig_norm, recon_norm, brain_mask, lpips_fn, device)

print("=" * 55)
print("  📊 복원 성능 평가 결과")
print("=" * 55)
print(f"  비교 파일 : {RECON_PATH}")
print(f"  ✔️ PSNR   : {current_psnr:7.3f} dB  (brain mask 내부)")
print(f"  ✔️ SNR    : {current_snr:7.3f} dB")
print(f"  ✔️ NRMSE  : {current_nrmse:.5f}")
print(f"  ✔️ SSIM   : {current_ssim:.5f}      (3축 평균)")
print(f"     ├ Axial    : {ssim_per_ax['axial']:.5f}")
print(f"     ├ Coronal  : {ssim_per_ax['coronal']:.5f}")
print(f"     └ Sagittal : {ssim_per_ax['sagittal']:.5f}")
print(f"  ✔️ LPIPS  : {current_lpips:.5f}")
print("=" * 55)