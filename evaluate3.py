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

# RECON_PATH = "003_flair_baseline.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_baseline.nii.gz"

# RECON_PATH = "recon_BraTS20_Training_003_flair.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_gate.nii.gz"

# RECON_PATH = "/dshome/ddualab/dongnyeok/Cunerf_0213/save/freq_08/BraTS20_Training_003_flair/eval/ours.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_cunerf_ex.nii.gz"

RECON_PATH = "003_flair_08.nii.gz"
ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_08.nii.gz"

# RECON_PATH = "003_flair_06_multiview.nii.gz"
# ALIGNED_RECON_SAVE_PATH = "003_flair_aligned_nesvor_06.nii.gz"


# ---------------------------------------------------------
# 2. 정렬 함수
#    ★ FIX 1: Case B — reorientation 후 shape 재확인
#    ★ FIX 2: 헤더 재사용 시 scl_slope/inter 초기화
# ---------------------------------------------------------
def robust_align_recon(orig_nib, recon_nib, save_path=None):
    """
    [Case A] Shape 동일, Orientation 동일  → affine 직접 복사
    [Case B] Shape 동일, Orientation 불일치 → 배열 재배치 후 affine 복사
    [Case C] Shape 불일치                  → canonical + resample
    """
    orig_axcodes  = nib.aff2axcodes(orig_nib.affine)
    recon_axcodes = nib.aff2axcodes(recon_nib.affine)
    orig_shape    = orig_nib.shape[:3]
    recon_shape   = recon_nib.shape[:3]

    print(f"[DEBUG] 원본   : orientation={orig_axcodes}, shape={orig_shape}")
    print(f"[DEBUG] 재구성 : orientation={recon_axcodes}, shape={recon_shape}")

    if orig_shape == recon_shape:

        if orig_axcodes == recon_axcodes:
            # Case A
            print("[INFO] Case A: Orientation 일치 → affine 직접 복사")
            recon_data = recon_nib.get_fdata().astype(np.float32)

        else:
            # Case B
            print(f"[INFO] Case B: Orientation 불일치 ({recon_axcodes}→{orig_axcodes}) "
                  "→ 배열 재배치 후 affine 복사")
            orig_ornt  = nib.io_orientation(orig_nib.affine)
            recon_ornt = nib.io_orientation(recon_nib.affine)
            transform  = nib.ornt_transform(recon_ornt, orig_ornt)
            recon_reoriented = recon_nib.as_reoriented(transform)
            recon_data = recon_reoriented.get_fdata().astype(np.float32)

            # ★ FIX 1: reorientation 후 shape이 달라지는 경우 방어
            if recon_data.shape[:3] != orig_shape:
                raise ValueError(
                    f"[ERROR] Reorientation 후 shape 불일치: "
                    f"{recon_data.shape[:3]} vs {orig_shape}.\n"
                    "       축 순서 변경을 수반하는 경우입니다. "
                    "Case C(resample) 방식으로 처리하거나 파일을 점검하세요."
                )

        # ★ FIX 2: 헤더 복사 후 스케일 관련 필드 초기화
        #   scl_slope/inter가 남아 있으면 다운스트림 툴에서 강도값이 왜곡될 수 있음
        new_header = orig_nib.header.copy()
        new_header.set_data_shape(recon_data.shape)
        new_header['scl_slope'] = 1.0
        new_header['scl_inter'] = 0.0
        recon_aligned = nib.Nifti1Image(recon_data, orig_nib.affine, new_header)
        orig_final = orig_nib

    else:
        # Case C
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
# 3. 정규화
#    ★ FIX 3: GT의 percentile로 두 볼륨을 동일한 척도로 정규화
#             (독립 정규화 → 밝기 오차가 소멸하는 문제 수정)
# ---------------------------------------------------------
def normalize_with_gt_params(orig_arr, recon_arr, mask):
    """
    GT(원본)의 0.1 / 99.9 percentile을 기준으로
    GT와 Recon 모두 동일한 [0, 1] 척도로 변환한다.

    핵심: Recon이 전체적으로 어둡거나 밝으면 정규화 후에도
          그 차이가 보존되어 PSNR/SNR에 반영된다.
    """
    mn = np.percentile(orig_arr[mask], 0.1)
    mx = np.percentile(orig_arr[mask], 99.9)
    print(f"  정규화 기준 (GT 기준) — 하한(0.1%ile): {mn:.2f}, 상한(99.9%ile): {mx:.2f}")

    def _apply(arr):
        out = np.clip((arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)
        out[~mask] = 0.0
        return out.astype(np.float32)

    return _apply(orig_arr), _apply(recon_arr)


# ---------------------------------------------------------
# 4. 평가 지표 함수
# ---------------------------------------------------------
def calculate_psnr_masked(ref, recon, mask):
    """
    ★ FIX 4: 전체 볼륨이 아닌 brain mask 내부 MSE로 PSNR 계산
              배경(0) 포함 시 오차가 희석되어 PSNR이 인위적으로 높아지는 문제 수정
    """
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
    """
    ★ NEW: Normalized RMSE
    PSNR/SNR과 다른 각도에서 오차를 측정; MRI 재구성 논문에서 표준적으로 함께 보고됨
    낮을수록 좋음
    """
    rmse        = np.sqrt(np.mean((ref[mask] - recon[mask]) ** 2))
    norm_factor = np.sqrt(np.mean(ref[mask] ** 2))
    return float(rmse / (norm_factor + 1e-8))


def calculate_ssim_3axes(ref_vol, recon_vol, mask_vol,
                          data_range=1.0, min_slice_voxels=100):
    """
    ★ FIX 5: Axial 단일 축 → 3축(Axial/Coronal/Sagittal) 슬라이스별 SSIM 평균
              3D SSIM 대비 메모리 효율적, 등방성 BraTS 볼륨 평가에 더 완전함
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
            if np.sum(mask_s) > min_slice_voxels:
                scores.append(ssim(ref_s, recon_s, data_range=data_range))
        per_axis[axis_name] = float(np.mean(scores)) if scores else 0.0

    mean_ssim = float(np.mean(list(per_axis.values())))
    return mean_ssim, per_axis


def calculate_lpips_3d(ref_vol, recon_vol, mask_vol,
                        loss_fn, device, min_slice_voxels=100):
    """
    ★ FIX 6: LPIPS 모델을 매 호출마다 초기화하지 않고 외부에서 받음 (비효율 수정)
              Axial 슬라이스 평균; [-1, 1] 범위로 변환 후 3채널 복제
    """
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
# 5. 메인
# ---------------------------------------------------------
print("=" * 55)
print("  데이터 로드 중...")
print("=" * 55)

orig_nib  = nib.load(ORIG_PATH)
recon_nib = nib.load(RECON_PATH)

# sform/qform 상태 확인 (정보용; 0이면 affine 신뢰도 낮음)
print(f"\n[헤더] 원본   sform_code={orig_nib.header['sform_code']}, "
      f"qform_code={orig_nib.header['qform_code']}")
print(f"[헤더] 재구성 sform_code={recon_nib.header['sform_code']}, "
      f"qform_code={recon_nib.header['qform_code']}")

# 정렬
print("\n정렬 중...")
orig_final, recon_aligned = robust_align_recon(
    orig_nib, recon_nib, save_path=ALIGNED_RECON_SAVE_PATH
)

orig_array  = orig_final.get_fdata().astype(np.float32)
recon_array = recon_aligned.get_fdata().astype(np.float32)

# ★ FIX 7: NaN/Inf 처리 (일부 재구성 모델 출력에 포함될 수 있음)
nan_count = np.sum(~np.isfinite(recon_array))
if nan_count > 0:
    print(f"[WARN] 재구성 볼륨에 NaN/Inf {nan_count:,}개 → 0으로 대체")
recon_array = np.nan_to_num(recon_array, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\n정렬 후 shape — 원본: {orig_array.shape}, 재구성: {recon_array.shape}")
assert orig_array.shape == recon_array.shape, "[ERROR] Shape 불일치! 정렬 실패."

# ★ FIX 8: brain_mask는 GT에서만 생성하여 두 볼륨에 동일 적용
#   - BraTS는 배경이 정확히 0으로 설정되어 있으므로 > 0이 신뢰할 수 있음
#   - GT 마스크 사용으로 모든 모델을 동일한 영역에서 공정하게 비교
brain_mask = orig_array > 0
print(f"brain_mask — GT 기준 생성, 유효 voxel 수: {np.sum(brain_mask):,}")

# 공유 정규화
print("\n정규화 중 (GT 기준 공유 정규화)...")
orig_norm, recon_norm = normalize_with_gt_params(
    orig_array, recon_array, brain_mask
)

# LPIPS 모델 1회 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n디바이스: {device}")
print("LPIPS 모델 로드 중 (AlexNet backbone)...")
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()


# ─────────────────────────────────────────────────
# ★ 선형 강도 보정 (추가 블록)
#   brain mask 내부에서 ||orig - k * recon||²를 최소화하는 k를 추정
# ─────────────────────────────────────────────────
k = (np.sum(orig_norm[brain_mask] * recon_norm[brain_mask]) /
     (np.sum(recon_norm[brain_mask] ** 2) + 1e-8))
print(f"\n[선형 강도 보정] 추정 스케일 k = {k:.4f}")
print(f"  k < 1 → 재구성이 GT보다 어두움")
print(f"  k > 1 → 재구성이 GT보다 밝음")

recon_norm_corrected = np.clip(recon_norm * k, 0.0, 1.0)
recon_norm_corrected[~brain_mask] = 0.0
# ─────────────────────────────────────────────────


# 보정 후 차이 맵 생성
diff_map = orig_norm - recon_norm_corrected
diff_map[~brain_mask] = 0

# 뇌 내부를 앞/뒤, 좌/우, 위/아래 8등분해서 구역별 평균 오차 확인
h, w, d = orig_norm.shape
for zi, zname in enumerate(['inf', 'sup']):
    for yi, yname in enumerate(['ant', 'pos']):
        for xi, xname in enumerate(['L', 'R']):
            region = diff_map[
                xi*h//2:(xi+1)*h//2,
                yi*w//2:(yi+1)*w//2,
                zi*d//2:(zi+1)*d//2
            ]
            mask_r = brain_mask[
                xi*h//2:(xi+1)*h//2,
                yi*w//2:(yi+1)*w//2,
                zi*d//2:(zi+1)*d//2
            ]
            if mask_r.sum() > 0:
                print(f"{xname}-{yname}-{zname}: mean_diff={region[mask_r].mean():.4f}")


# 지표 계산
print("지표 계산 중...\n")
current_psnr               = calculate_psnr_masked(orig_norm, recon_norm_corrected, brain_mask)
current_snr                = calculate_snr(orig_norm, recon_norm_corrected, brain_mask)
current_nrmse              = calculate_nrmse(orig_norm, recon_norm_corrected, brain_mask)
current_ssim, ssim_per_ax  = calculate_ssim_3axes(orig_norm, recon_norm_corrected, brain_mask)
current_lpips              = calculate_lpips_3d(
    orig_norm, recon_norm_corrected, brain_mask, lpips_fn, device
)

print("=" * 55)
print("  📊 복원 성능 평가 결과")
print("=" * 55)
print(f"  비교 파일 : {RECON_PATH}  [선형 강도 보정 적용, k={k:.4f}]")
print(f"  ✔️ PSNR   : {current_psnr:7.3f} dB  (brain mask 내부)")
print(f"  ✔️ SNR    : {current_snr:7.3f} dB")
print(f"  ✔️ NRMSE  : {current_nrmse:.5f}      (낮을수록 좋음)")
print(f"  ✔️ SSIM   : {current_ssim:.5f}      (3축 평균)")
print(f"     ├ Axial    : {ssim_per_ax['axial']:.5f}")
print(f"     ├ Coronal  : {ssim_per_ax['coronal']:.5f}")
print(f"     └ Sagittal : {ssim_per_ax['sagittal']:.5f}")
print(f"  ✔️ LPIPS  : {current_lpips:.5f}      (Axial, AlexNet, 낮을수록 좋음)")
print("=" * 55)
print("\n[진단] 정규화 전 원본(GT)과 재구성 볼륨의 실제 스케일 비교")
print(f" - GT    | 평균: {orig_array[brain_mask].mean():.4f}, 최대값: {orig_array[brain_mask].max():.4f}")
print(f" - Recon | 평균: {recon_array[brain_mask].mean():.4f}, 최대값: {recon_array[brain_mask].max():.4f}")
print("-" * 55)
