#!/usr/bin/env bash
# Fixed-setting comparison on five TEST subjects; do not use for tuning.
# Run from the activated nesvor environment. Evaluation uses nerf_tcnn Python.
# Usage: bash run_ff_coordinate_comparison.sh
# Preview: DRY_RUN=1 bash run_ff_coordinate_comparison.sh
# Evaluation only: SKIP_TRAIN=1 bash run_ff_coordinate_comparison.sh
# Include a freshly reconstructed no-FF control: RUN_NO_FF=1 bash ...
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dshome/ddualab/dongnyeok/NeSVoR_alt_IC}
OLD_RECON_ROOT=${OLD_RECON_ROOT:-/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s}
DEGRADED_ROOT=${DEGRADED_ROOT:-/data/BraTS20_Degraded_4x_5}
WORK_ROOT=${WORK_ROOT:-${PROJECT_ROOT}/ff_coordinate_comparison_5cases}
EVAL_SCRIPT=${EVAL_SCRIPT:-/dshome/ddualab/dongnyeok/NeSVoR/evaluate_batch_v2.py}
TRAIN_PYTHON=${TRAIN_PYTHON:-python}
EVAL_PYTHON=${EVAL_PYTHON:-/dshome/ddualab/conda_env/nerf_tcnn/bin/python}
DRY_RUN=${DRY_RUN:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
RUN_NO_FF=${RUN_NO_FF:-0}
DEVICE=${DEVICE:-0}
INITIAL_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
TRAIN_LD_LIBRARY_PATH=${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${INITIAL_LD_LIBRARY_PATH}
EVAL_LD_LIBRARY_PATH=${EVAL_PYTHON%/bin/python}/lib:${INITIAL_LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}

IDS=(BraTS20_Training_003 BraTS20_Training_030 BraTS20_Training_034
     BraTS20_Training_040 BraTS20_Training_060)
MODALITIES=(flair t1ce)
FIXED_TAG=G14s_ff_coord_fixed
NO_FF_TAG=G14s_ff_off
FIXED_ROOT=${WORK_ROOT}/recon_fixed
NO_FF_ROOT=${WORK_ROOT}/recon_no_ff
CASE_CSV=${WORK_ROOT}/patients_5_test.csv
RESULT_CSV=${WORK_ROOT}/metrics_all.csv
PAIR_CSV=${WORK_ROOT}/paired_differences.csv
SUMMARY_CSV=${WORK_ROOT}/mean_differences.csv
COMMON_ARGS=(
  --output-resolution 1.0 --output-intensity-mean 308
  --no-transformation-optimization --registration none --single-precision
  --patch-size 48 --n-patches 8 --weight-image 1.0 --delta 0.05
  --diversity-loss-fn variance --diversity-loss-space raw --target-diversity-var 0.06
  --n-iter 2000 --batch-size 4096 --learning-rate 5e-3 --gamma 0.33
  --milestones 0.5 0.75 0.9 --seed 42 --device "${DEVICE}"
  --ff-alpha 2.0 --spatial-gating --gating-hidden-dim 8
  --gating-activation silu --gating-grad-clip 0.1 --gating-lr-scale 2.0
)
# Load the console entry using TRAIN_PYTHON, avoiding a mismatched nesvor executable.
CLI_LOADER='import sys; from importlib import metadata; e=metadata.entry_points(); m=list(e.select(group="console_scripts",name="nesvor")) if hasattr(e,"select") else [x for x in e.get("console_scripts",[]) if x.name=="nesvor"]; assert len(m)==1,"Expected exactly one nesvor console entry"; sys.argv=["nesvor"]+sys.argv[1:]; m[0].load()()'

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
show_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
for flag in DRY_RUN SKIP_TRAIN SKIP_EVAL RUN_NO_FF; do
  [[ ${!flag} == 0 || ${!flag} == 1 ]] || die "${flag} must be 0 or 1"
done
printf 'Five TEST subjects; fixed settings, seed 42; modalities: %s\n' "${MODALITIES[*]}"
printf 'Old reconstructions: %s\nNew outputs: %s\n' "$OLD_RECON_ROOT" "$WORK_ROOT"

if [[ $DRY_RUN == 0 ]]; then
  [[ -d $PROJECT_ROOT ]] || die "Project missing: $PROJECT_ROOT"
  cd "$PROJECT_ROOT"
  command -v "$TRAIN_PYTHON" >/dev/null || die "Activate the nesvor environment"
  if [[ $SKIP_EVAL == 0 ]]; then
    [[ -x $EVAL_PYTHON ]] || die "Evaluation Python missing: $EVAL_PYTHON"
    [[ -f $EVAL_SCRIPT ]] || die "Evaluation script missing; set EVAL_SCRIPT=/absolute/path/evaluate_batch_v2.py"
  fi
  # Verify inputs before launching any reconstruction.
  for id in "${IDS[@]}"; do
    short=${id##*_}
    for mod in "${MODALITIES[@]}"; do
      if [[ $SKIP_TRAIN == 0 ]]; then
        [[ -s ${DEGRADED_ROOT}/${id}/${id}_${mod}.nii ]] || die "Degraded input missing: ${id}/${id}_${mod}.nii"
      fi
      if [[ $SKIP_EVAL == 0 ]]; then
        [[ -s ${OLD_RECON_ROOT}/${id}/${short}_${mod}_4x5_G14s.nii.gz ]] || die "Old reconstruction missing: ${id}/${mod}"
        [[ -s /data/BraTS2020_TrainingData/${id}/${id}_${mod}.nii || -s /data/BraTS2020_TrainingData/${id}/${id}_${mod}.nii.gz ]] || die "GT missing: ${id}/${mod}"
      fi
    done
  done
  mkdir -p "$WORK_ROOT/logs"
  printf 'Brats20ID\n' > "$CASE_CSV"
  printf '%s\n' "${IDS[@]}" >> "$CASE_CSV"
  # Record the actual editable-install source and verify the scaling fix.
  env LD_LIBRARY_PATH="$TRAIN_LD_LIBRARY_PATH" "$TRAIN_PYTHON" - "$PROJECT_ROOT" <<'PY' | tee "$WORK_ROOT/logs/installed_source.txt"
import ast, hashlib, importlib, inspect, os, sys, textwrap
root = os.path.realpath(sys.argv[1])
module = importlib.import_module('nesvor.inr.models')
path = os.path.realpath(module.__file__)
print('Training Python:', sys.executable)
print('Installed models:', path)
if os.path.commonpath([root, path]) != root:
    raise SystemExit('Editable install points to another project. Correct PROJECT_ROOT or the environment.')
tree = ast.parse(textwrap.dedent(inspect.getsource(module.NeSVoR.patch_forward)))
fixed = any(isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'xyz_flat' for t in n.targets)
            and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div)
            and isinstance(n.value.right, ast.Attribute)
            and isinstance(n.value.right.value, ast.Name) and n.value.right.value.id == 'self'
            and n.value.right.attr == 'spatial_scaling' for n in ast.walk(tree))
if not fixed:
    raise SystemExit('Expected xyz_flat = xyz_patch.view(...) / self.spatial_scaling. Fix patch_forward first.')
print('Coordinate scaling fix: present')
print('models.py SHA256:', hashlib.sha256(open(path, 'rb').read()).hexdigest())
print('These are test subjects: compare fixed settings; do not tune on their results.')
PY
fi

run_case() {
  local id=$1 mod=$2 root=$3 tag=$4 ff_weight=$5
  local short=${id##*_} input=${DEGRADED_ROOT}/${id}/${id}_${mod}.nii
  local folder=${root}/${id}
  local volume=${folder}/${short}_${mod}_4x5_${tag}.nii.gz
  local model=${folder}/${short}_${mod}_4x5_${tag}.pt
  local done_marker=${folder}/${short}_${mod}_4x5_${tag}.complete
  local logfile=${WORK_ROOT}/logs/${short}_${mod}_${tag}.log
  local -a cmd=(env "LD_LIBRARY_PATH=$TRAIN_LD_LIBRARY_PATH" "$TRAIN_PYTHON" -c "$CLI_LOADER"
    reconstruct --input-stacks "$input" --sample-mask "$input"
    --output-volume "$volume" --output-model "$model"
    "${COMMON_ARGS[@]}" --weight-ff-loss "$ff_weight")
  if [[ $DRY_RUN == 1 ]]; then
    show_command "${cmd[@]}"
    return
  fi
  if [[ -f $done_marker && -s $volume && -s $model ]]; then
    printf '[SKIP completed] %s / %s / %s\n' "$tag" "$id" "$mod"
    return
  fi
  [[ ! -e $volume && ! -e $model && ! -e $done_marker ]] || die "Incomplete output exists at $folder for $mod. Move that case's files aside before retrying."
  mkdir -p "$folder"
  printf '[START] %s / %s / %s\n' "$tag" "$id" "$mod"
  show_command "${cmd[@]}" > "${logfile%.log}.command.txt"
  if "${cmd[@]}" 2>&1 | tee "$logfile"; then
    [[ -s $volume && -s $model ]] || die "Command returned success but outputs are missing: $id/$mod"
    touch "$done_marker"
  else
    die "Reconstruction failed: $tag/$id/$mod. See $logfile"
  fi
}

if [[ $SKIP_TRAIN == 0 ]]; then
  for id in "${IDS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
      run_case "$id" "$mod" "$FIXED_ROOT" "$FIXED_TAG" 0.05
      if [[ $RUN_NO_FF == 1 ]]; then
        run_case "$id" "$mod" "$NO_FF_ROOT" "$NO_FF_TAG" 0.0
      fi
    done
  done
fi

if [[ $SKIP_EVAL == 0 ]]; then
  recon_dirs=("${OLD_RECON_ROOT}:G14s_original" "${FIXED_ROOT}:${FIXED_TAG}")
  if [[ $RUN_NO_FF == 1 ]]; then recon_dirs+=("${NO_FF_ROOT}:${NO_FF_TAG}"); fi
  # The supplied evaluator ignores degraded-root and uses its hardcoded GT_ROOT.
  eval_cmd=(env "LD_LIBRARY_PATH=$EVAL_LD_LIBRARY_PATH" "$EVAL_PYTHON" "$EVAL_SCRIPT"
    --recon-dirs "${recon_dirs[@]}" --test-csv "$CASE_CSV"
    --modalities "${MODALITIES[@]}" --output-csv "$RESULT_CSV")
  if [[ $DRY_RUN == 1 ]]; then
    show_command "${eval_cmd[@]}"
    exit 0
  fi
  for id in "${IDS[@]}"; do
    short=${id##*_}
    for mod in "${MODALITIES[@]}"; do
      [[ -s ${FIXED_ROOT}/${id}/${short}_${mod}_4x5_${FIXED_TAG}.nii.gz ]] || die "Fixed reconstruction missing: $id/$mod"
      if [[ $RUN_NO_FF == 1 ]]; then
        [[ -s ${NO_FF_ROOT}/${id}/${short}_${mod}_4x5_${NO_FF_TAG}.nii.gz ]] || die "No-FF reconstruction missing: $id/$mod"
      fi
    done
  done
  show_command "${eval_cmd[@]}" > "$WORK_ROOT/logs/evaluation.command.txt"
  "${eval_cmd[@]}" 2>&1 | tee "$WORK_ROOT/logs/evaluation.log"
  # The evaluator catches case errors and can exit 0 with blank metrics.
  # Validate completeness, then report paired differences without statistical tests.
  "$EVAL_PYTHON" - "$RESULT_CSV" "$CASE_CSV" "$PAIR_CSV" "$SUMMARY_CSV" "$RUN_NO_FF" <<'PY'
import csv, math, sys
raw, cases, paired_path, summary_path, no_ff = sys.argv[1:]
with open(cases, newline='') as f: ids = [r['Brats20ID'] for r in csv.DictReader(f)]
methods = ['G14s_original', 'G14s_ff_coord_fixed']
if no_ff == '1': methods.append('G14s_ff_off')
metrics = ['PSNR', 'SNR', 'NRMSE', 'SSIM', 'LPIPS', 'VIF', 'FSIM', 'HFEN']
lower = {'NRMSE', 'LPIPS', 'HFEN'}
with open(raw, newline='') as f: rows = list(csv.DictReader(f))
indexed = {}
for row in rows:
    key = (row['patient_id'], row['modality'], row['method'])
    if key in indexed: raise SystemExit(f'Duplicate evaluation row: {key}')
    indexed[key] = row
expected = {(i, m, t) for i in ids for m in ('flair', 't1ce') for t in methods}
if set(indexed) != expected:
    raise SystemExit(f'Evaluation keys differ: missing={expected-set(indexed)}, extra={set(indexed)-expected}')
for key, row in indexed.items():
    for metric in metrics:
        val = row.get(metric, '')
        if not val and metric == 'FSIM': continue  # evaluator allows absent piq
        try: value = float(val)
        except (TypeError, ValueError): raise SystemExit(f'Missing/invalid {metric} for {key}; inspect evaluation.log')
        if not math.isfinite(value): raise SystemExit(f'Nonfinite {metric} for {key}; inspect evaluation.log')
comparisons = [('G14s_original', 'G14s_ff_coord_fixed')]
if no_ff == '1': comparisons.append(('G14s_ff_off', 'G14s_ff_coord_fixed'))
paired = []
for baseline, target in comparisons:
    for patient in ids:
        for mod in ('flair', 't1ce'):
            a, b = indexed[(patient, mod, baseline)], indexed[(patient, mod, target)]
            for metric in metrics:
                if not a[metric] or not b[metric]: continue
                av, bv = float(a[metric]), float(b[metric])
                paired.append(dict(patient_id=patient, modality=mod, baseline=baseline,
                    target=target, metric=metric, baseline_value=av, target_value=bv,
                    delta_target_minus_baseline=bv-av,
                    improvement=av-bv if metric in lower else bv-av))
fields = ['patient_id','modality','baseline','target','metric','baseline_value',
          'target_value','delta_target_minus_baseline','improvement']
with open(paired_path, 'w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(paired)
summary=[]
for baseline, target in comparisons:
    for mod in ('flair','t1ce','both_patient_mean'):
        for metric in metrics:
            subset=[r for r in paired if r['baseline']==baseline and r['target']==target
                    and r['metric']==metric and (mod=='both_patient_mean' or r['modality']==mod)]
            # For both modalities, average within each patient first.
            by_patient={}
            for r in subset: by_patient.setdefault(r['patient_id'], []).append(r)
            complete=[v for v in by_patient.values() if len(v)==(2 if mod=='both_patient_mean' else 1)]
            if not complete: continue
            def avg(field):
                return sum(sum(r[field] for r in v)/len(v) for v in complete)/len(complete)
            summary.append(dict(baseline=baseline,target=target,modality=mod,metric=metric,
                n_patients=len(complete),baseline_mean=avg('baseline_value'),
                target_mean=avg('target_value'),mean_delta=avg('delta_target_minus_baseline'),
                mean_improvement=avg('improvement')))
with open(summary_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['baseline','target','modality','metric','n_patients',
        'baseline_mean','target_mean','mean_delta','mean_improvement'])
    w.writeheader(); w.writerows(summary)
print('\nMean differences (positive improvement = better for every metric):')
for r in summary:
    if r['metric'] in ('PSNR','SSIM','LPIPS'):
        print(f"{r['baseline']} -> {r['target']} | {r['modality']:17s} | {r['metric']:5s} | "
              f"{r['baseline_mean']:.6g} -> {r['target_mean']:.6g} | improvement={r['mean_improvement']:+.6g}")
print('Paired differences:', paired_path)
print('Mean differences:', summary_path)
if any(not r.get('FSIM') for r in rows): print('FSIM incomplete (piq may be absent); no complete FSIM comparison where values are missing.')
PY
fi
printf '\nCompleted. Outputs: %s\n' "$WORK_ROOT"
