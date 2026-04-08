import os
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
import lpips
from skimage.metrics import structural_similarity as ssim


def robust_align_recon(orig_nib, recon_nib, save_path=None):
    orig_shape = orig_nib.shape[:3]
    orig_ornt = nib.io_orientation(orig_nib.affine)
    recon_ornt = nib.io_orientation(recon_nib.affine)

    if not np.array_equal(orig_ornt, recon_ornt):
        transform = nib.ornt_transform(recon_ornt, orig_ornt)
        recon_nib = recon_nib.as_reoriented(transform)

    recon_shape = recon_nib.shape[:3]
    recon_data = recon_nib.get_fdata().astype(np.float32)

    if orig_shape != recon_shape:
        inv_orig_affine = np.linalg.inv(orig_nib.affine)
        recon_origin_physical = recon_nib.affine @ np.array([0, 0, 0, 1])
        offset_voxels = inv_orig_affine @ recon_origin_physical
        x_off, y_off, z_off = np.round(offset_voxels[:3]).astype(int)

        aligned_data = np.zeros(orig_shape, dtype=np.float32)
        rx, ry, rz = recon_shape

        tgt_x_s = max(0, x_off)
        tgt_x_e = min(orig_shape[0], x_off + rx)
        tgt_y_s = max(0, y_off)
        tgt_y_e = min(orig_shape[1], y_off + ry)
        tgt_z_s = max(0, z_off)
        tgt_z_e = min(orig_shape[2], z_off + rz)

        src_x_s = max(0, -x_off)
        src_y_s = max(0, -y_off)
        src_z_s = max(0, -z_off)

        src_x_e = src_x_s + (tgt_x_e - tgt_x_s)
        src_y_e = src_y_s + (tgt_y_e - tgt_y_s)
        src_z_e = src_z_s + (tgt_z_e - tgt_z_s)

        if (tgt_x_e > tgt_x_s) and (tgt_y_e > tgt_y_s) and (tgt_z_e > tgt_z_s):
            aligned_data[tgt_x_s:tgt_x_e, tgt_y_s:tgt_y_e, tgt_z_s:tgt_z_e] = \
                recon_data[src_x_s:src_x_e, src_y_s:src_y_e, src_z_s:src_z_e]

        recon_data = aligned_data

    new_header = orig_nib.header.copy()
    new_header.set_data_shape(recon_data.shape)
    new_header["scl_slope"] = 1.0
    new_header["scl_inter"] = 0.0
    recon_aligned = nib.Nifti1Image(recon_data, orig_nib.affine, new_header)

    if save_path is not None:
        nib.save(recon_aligned, save_path)

    return orig_nib, recon_aligned


def calculate_psnr(ref, recon, mask):
    mse = np.mean((ref[mask] - recon[mask]) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def calculate_nrmse(ref, recon, mask):
    rmse = np.sqrt(np.mean((ref[mask] - recon[mask]) ** 2))
    norm_factor = np.sqrt(np.mean(ref[mask] ** 2))
    return float(rmse / (norm_factor + 1e-8))


def calculate_ssim_3axes(ref_vol, recon_vol, mask_vol, data_range=1.0, min_slice_voxels=100):
    axes = {"axial": 2, "coronal": 1, "sagittal": 0}
    per_axis = {}

    for axis_name, axis in axes.items():
        scores = []
        for i in range(ref_vol.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i
            ref_s = ref_vol[tuple(idx)]
            recon_s = recon_vol[tuple(idx)]
            mask_s = mask_vol[tuple(idx)]

            if np.sum(mask_s) > min_slice_voxels:
                scores.append(ssim(ref_s, recon_s, data_range=data_range))

        per_axis[axis_name] = float(np.mean(scores)) if scores else 0.0

    return float(np.mean(list(per_axis.values())))


def calculate_lpips_3d(ref_vol, recon_vol, mask_vol, loss_fn, device, min_slice_voxels=100):
    scores = []

    for i in range(ref_vol.shape[2]):
        if np.sum(mask_vol[:, :, i]) > min_slice_voxels:
            r = torch.from_numpy(ref_vol[:, :, i] * 2.0 - 1.0).float()
            rc = torch.from_numpy(recon_vol[:, :, i] * 2.0 - 1.0).float()

            r = r.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            rc = rc.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

            with torch.no_grad():
                scores.append(loss_fn(r, rc).item())

    return float(np.mean(scores)) if scores else 0.0


def evaluate_case(patient, modality, exp_tag, gt_root, recon_dir, aligned_dir, lpips_fn, device, save_aligned):
    gt_path = os.path.join(
        gt_root,
        f"BraTS20_Training_{patient}",
        f"BraTS20_Training_{patient}_{modality}.nii",
    )
    recon_path = os.path.join(recon_dir, f"{patient}_{modality}_4x5_{exp_tag}.nii.gz")

    if save_aligned:
        os.makedirs(aligned_dir, exist_ok=True)
        aligned_save = os.path.join(aligned_dir, f"{patient}_{modality}_4x5_{exp_tag}_aligned.nii.gz")
    else:
        aligned_save = None

    print(f"\n[{exp_tag}] [{patient} / {modality}] 평가 시작")

    if not os.path.exists(gt_path):
        print(f"  ⚠️ GT 파일 없음: {gt_path} → SKIP")
        return None

    if not os.path.exists(recon_path):
        print(f"  ⚠️ 재구성 파일 없음: {recon_path} → SKIP")
        return None

    orig_nib = nib.load(gt_path)
    recon_nib = nib.load(recon_path)
    _, recon_aligned = robust_align_recon(orig_nib, recon_nib, save_path=aligned_save)

    orig_arr = orig_nib.get_fdata().astype(np.float32)
    recon_arr = recon_aligned.get_fdata().astype(np.float32)
    recon_arr = np.nan_to_num(recon_arr, nan=0.0, posinf=0.0, neginf=0.0)

    brain_mask = orig_arr > 0
    if np.sum(brain_mask) == 0:
        print("  ⚠️ brain mask가 비어 있음 → SKIP")
        return None

    k = np.sum(orig_arr[brain_mask] * recon_arr[brain_mask]) / (
        np.sum(recon_arr[brain_mask] ** 2) + 1e-8
    )
    recon_arr = recon_arr * k

    mn = np.percentile(orig_arr[brain_mask], 0.1)
    mx = np.percentile(orig_arr[brain_mask], 99.9)

    orig_norm = np.clip((orig_arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    recon_norm = np.clip((recon_arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)

    orig_norm[~brain_mask] = 0.0
    recon_norm[~brain_mask] = 0.0

    psnr = calculate_psnr(orig_norm, recon_norm, brain_mask)
    ssim_ = calculate_ssim_3axes(orig_norm, recon_norm, brain_mask)
    nrmse = calculate_nrmse(orig_norm, recon_norm, brain_mask)
    lpips_score = calculate_lpips_3d(orig_norm, recon_norm, brain_mask, lpips_fn, device)

    print(f"  PSNR={psnr:.3f}  SSIM={ssim_:.5f}  NRMSE={nrmse:.5f}  LPIPS={lpips_score:.5f}")

    return {
        "exp_tag": exp_tag,
        "patient": patient,
        "modality": modality,
        "PSNR": psnr,
        "SSIM": ssim_,
        "NRMSE": nrmse,
        "LPIPS": lpips_score,
    }


def save_csv(results, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["exp_tag", "patient", "modality", "PSNR", "SSIM", "NRMSE", "LPIPS"]
        )
        writer.writeheader()
        writer.writerows(results)


def print_single_tag_summary(results, exp_tag, patients, modalities):
    res_map = {(r["patient"], r["modality"]): r for r in results}
    metrics = ["PSNR", "SSIM", "LPIPS"]

    print("\n" + "=" * 100)
    print(f"  실험: {exp_tag}  —  평가 결과 요약")
    print("=" * 100)
    print(f"{'':>5}  " + "  ".join(f"{'[' + mod + ']':^27}" for mod in modalities))
    print(f"{'':>5}  " + "  ".join("  ".join(f"{m:<7}" for m in metrics) for _ in modalities))
    print("-" * 100)

    for p in patients:
        row = f"{p:>5}  "
        for mod in modalities:
            key = (p, mod)
            if key in res_map:
                r = res_map[key]
                row += f"{r['PSNR']:7.3f}  {r['SSIM']:.5f}  {r['LPIPS']:.5f}  "
            else:
                row += f"{'N/A':>7}  {'N/A':>7}  {'N/A':>7}  "
        print(row)

    print("-" * 100)

    avg_row = f"{'AVG':>5}  "
    for mod in modalities:
        mod_vals = [res_map[(p, mod)] for p in patients if (p, mod) in res_map]
        if mod_vals:
            avg_psnr = np.mean([v["PSNR"] for v in mod_vals])
            avg_ssim = np.mean([v["SSIM"] for v in mod_vals])
            avg_lpips = np.mean([v["LPIPS"] for v in mod_vals])
            avg_row += f"{avg_psnr:7.3f}  {avg_ssim:.5f}  {avg_lpips:.5f}  "
        else:
            avg_row += f"{'N/A':>7}  {'N/A':>7}  {'N/A':>7}  "
    print(avg_row)
    print("=" * 100)

    if results:
        overall_psnr = np.mean([r["PSNR"] for r in results])
        overall_ssim = np.mean([r["SSIM"] for r in results])
        overall_lpips = np.mean([r["LPIPS"] for r in results])
        print(
            f"\n  전체 평균 ({len(results)} cases)  "
            f"PSNR={overall_psnr:.3f}  SSIM={overall_ssim:.5f}  LPIPS={overall_lpips:.5f}"
        )
        print("=" * 100)


def print_multi_tag_comparison(all_results, exp_tags, patients, modalities):
    print("\n" + "=" * 100)
    print("  실험 간 평균 비교")
    print("=" * 100)
    print(f"{'TAG':<8} {'cases':>5}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
    print("-" * 100)

    for tag in exp_tags:
        tag_res = [r for r in all_results if r["exp_tag"] == tag]
        if not tag_res:
            print(f"{tag:<8} {'0':>5}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
            continue

        overall_psnr = np.mean([r["PSNR"] for r in tag_res])
        overall_ssim = np.mean([r["SSIM"] for r in tag_res])
        overall_lpips = np.mean([r["LPIPS"] for r in tag_res])

        print(f"{tag:<8} {len(tag_res):>5}  {overall_psnr:8.3f}  {overall_ssim:8.5f}  {overall_lpips:8.5f}")

    print("=" * 100)

    for mod in modalities:
        print(f"\n[{mod}]")
        print(f"{'TAG':<8} {'cases':>5}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
        print("-" * 60)
        for tag in exp_tags:
            vals = [r for r in all_results if r["exp_tag"] == tag and r["modality"] == mod]
            if not vals:
                print(f"{tag:<8} {'0':>5}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
                continue

            print(
                f"{tag:<8} {len(vals):>5}  "
                f"{np.mean([v['PSNR'] for v in vals]):8.3f}  "
                f"{np.mean([v['SSIM'] for v in vals]):8.5f}  "
                f"{np.mean([v['LPIPS'] for v in vals]):8.5f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=str, default="/data/BraTS2020_TrainingData")
    parser.add_argument("--recon-dir", type=str, default=".")
    parser.add_argument("--aligned-dir", type=str, default="./aligned_eval")
    parser.add_argument("--output-csv", type=str, default="eval_sweep_results.csv")
    parser.add_argument("--exp-tags", nargs="+", default=["H1", "H2", "H3", "E3"])
    parser.add_argument("--patients", nargs="+", default=["003", "026", "030", "040", "060"])
    parser.add_argument("--modalities", nargs="+", default=["flair", "t2", "t1ce"])
    parser.add_argument("--save-aligned", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LPIPS 모델 로드 중... device={device}")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    all_results = []

    for exp_tag in args.exp_tags:
        tag_results = []

        for patient in args.patients:
            for modality in args.modalities:
                res = evaluate_case(
                    patient=patient,
                    modality=modality,
                    exp_tag=exp_tag,
                    gt_root=args.gt_root,
                    recon_dir=args.recon_dir,
                    aligned_dir=args.aligned_dir,
                    lpips_fn=lpips_fn,
                    device=device,
                    save_aligned=args.save_aligned,
                )
                if res is not None:
                    all_results.append(res)
                    tag_results.append(res)

        print_single_tag_summary(tag_results, exp_tag, args.patients, args.modalities)

    if not all_results:
        print("평가된 결과가 없습니다.")
        raise SystemExit(1)

    save_csv(all_results, args.output_csv)
    print(f"\n결과 저장 완료: {args.output_csv}")

    print_multi_tag_comparison(all_results, args.exp_tags, args.patients, args.modalities)


if __name__ == "__main__":
    main()
