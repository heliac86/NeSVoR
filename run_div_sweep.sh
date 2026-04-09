#!/usr/bin/env bash
# ================================================================
# run_div_sweep.sh  (v1 — 2026-04-09)
# Diversity Loss 하이퍼파라미터 스윕
# (--weight-diversity-loss × --target-diversity-var 교호작용 탐색)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# HM 스윕 결과:
#   G5_repro  : seed 미고정 재현 기준선
#   HM2       : +0.055 dB (seed 노이즈 범위 내 → seed 42 재검증 필요)
#   warmup    : -1.605 dB (완전 기각)
#
# 현재 G5_lowvar BASE:
#   --weight-diversity-loss 0.05  (loss 강도 스케일)
#   --target-diversity-var  0.12  (loss 활성화 임계점)
#
# 교호작용 구조:
#   target_var = loss가 켜지는 임계점.
#                var(level_weights) < target_var 일 때만 loss 발생.
#                높을수록 loss가 더 오래, 더 강하게 작동.
#   weight     = loss의 크기 스케일.
#
#   target 높음 + weight 높음 → level_weights 과도 분화 (발산 위험)
#   target 높음 + weight 낮음 → 천천히, 안정적으로 분화
#   target 낮음 + weight 높음 → loss가 일찍 꺼지고 강하게 밀어붙임
#   target 낮음 + weight 낮음 → 분화 압력 거의 없음
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q0. seed 42 고정 시 G5_lowvar 기준선은 얼마인가?
#     → G5_lowvar_s42
# Q1. seed 42 고정 시 HM2 신호가 유의미하게 재현되는가?
#     → HM2_s42 vs G5_lowvar_s42
# Q2. target을 0.06으로 낮추면 성능이 오르는가?
#     (현재 0.12가 과도한 분화 요구인지 확인)
#     → div_lowtarget vs G5_lowvar_s42
# Q3. weight만 0.02로 낮추면 어떻게 되는가?
#     (target 유지, 분화 압력만 줄이기)
#     → div_lowweight vs G5_lowvar_s42
# Q4. 둘 다 강하게 (weight=0.08, target=0.15) 올리면 개선되는가?
#     (아직 분화가 부족하다는 가설 검증)
#     → div_stronger vs G5_lowvar_s42
# Q5. weight 높이되 target 낮추면 어떻게 되는가?
#     (강하게 밀되 일찍 멈추는 조합)
#     → div_tighter vs G5_lowvar_s42
#
# ── 실험 세트 ──────────────────────────────────────────────────
#   [0] G5_lowvar_s42  : BASE_ARGS + seed 42 고정 (새 기준선)
#   [1] HM2_s42        : BASE_ARGS + seed 42 + hard-mining-main-loss
#   [2] div_lowtarget  : weight=0.05, target=0.06
#   [3] div_lowweight  : weight=0.02, target=0.12
#   [4] div_stronger   : weight=0.08, target=0.15
#   [5] div_tighter    : weight=0.08, target=0.06
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 6세트 x 15 = 90
#
# 사용법:
#   chmod +x run_div_sweep.sh
#   ./run_div_sweep.sh                              # 모든 세트 실행
#   ENABLE_G5_LOWVAR_S42=0  ./run_div_sweep.sh      # 기준선 건너뜀
#   ENABLE_HM2_S42=0        ./run_div_sweep.sh      # HM2_s42 건너뜀
#   ENABLE_DIV_LOWTARGET=0  ./run_div_sweep.sh      # div_lowtarget 건너뜀
#   ENABLE_DIV_LOWWEIGHT=0  ./run_div_sweep.sh      # div_lowweight 건너뜀
#   ENABLE_DIV_STRONGER=0   ./run_div_sweep.sh      # div_stronger 건너뜀
#   ENABLE_DIV_TIGHTER=0    ./run_div_sweep.sh      # div_tighter 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_G5_LOWVAR_S42=${ENABLE_G5_LOWVAR_S42:-1}
ENABLE_HM2_S42=${ENABLE_HM2_S42:-1}
ENABLE_DIV_LOWTARGET=${ENABLE_DIV_LOWTARGET:-1}
ENABLE_DIV_LOWWEIGHT=${ENABLE_DIV_LOWWEIGHT:-1}
ENABLE_DIV_STRONGER=${ENABLE_DIV_STRONGER:-1}
ENABLE_DIV_TIGHTER=${ENABLE_DIV_TIGHTER:-1}

# ── BASE_ARGS: G5_lowvar 완전 재현 인자 ───────────────────────
# 모든 세트는 이 값을 기준으로 ±변경만 적용.
# seed 42 지정
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-ff-loss 0.10
  --ff-alpha 1.0
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.05
  --target-diversity-var 0.12
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
  --seed 42
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t2 t1ce)

FAILED_CASES=()
TOTAL=0
SUCCESS=0

# ── 단일 케이스 실행 함수 ──────────────────────────────────────
run_case() {
    local EXP_TAG=$1
    local PATIENT=$2
    local MODALITY=$3
    shift 3
    local EXTRA_ARGS="$*"

    local INPUT="${DEGRADED_ROOT}/BraTS20_Training_${PATIENT}/BraTS20_Training_${PATIENT}_${MODALITY}.nii"
    local OUTPUT_VOL="${PATIENT}_${MODALITY}_4x5_${EXP_TAG}.nii.gz"
    local OUTPUT_MODEL="${PATIENT}_${MODALITY}_${EXP_TAG}_model.pt"

    echo ""
    echo "================================================================"
    echo "  [START] Tag=${EXP_TAG}  Patient=${PATIENT}  Modality=${MODALITY}"
    echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # shellcheck disable=SC2086
    nesvor reconstruct \
        --input-stacks  "${INPUT}" \
        --output-volume "${OUTPUT_VOL}" \
        --output-model  "${OUTPUT_MODEL}" \
        --sample-mask   "${INPUT}" \
        --output-intensity-mean ${INTENSITY_MEAN} \
        ${EXTRA_ARGS}

    local EXIT_CODE=$?
    TOTAL=$((TOTAL + 1))
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ FAILED: ${EXP_TAG} / ${PATIENT} / ${MODALITY}  (exit=${EXIT_CODE})"
        FAILED_CASES+=("${EXP_TAG}_${PATIENT}_${MODALITY}")
    else
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ DONE : ${EXP_TAG} / ${PATIENT} / ${MODALITY}"
        echo "  Vol   → ${OUTPUT_VOL}"
        echo "  Model → ${OUTPUT_MODEL}"
        echo "  Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# ================================================================
# 세트 0: G5_lowvar_s42  (새 기준선)
#
#   목적: seed 42 고정 시 G5_lowvar의 기준 수치 확정.
#         이후 모든 div 탐색 실험의 비교 기준이 됨.
#         seed 미고정 결과(PSNR 25.664)와의 차이도 확인.
#   변경점: + --seed 42
# ================================================================
if [ "${ENABLE_G5_LOWVAR_S42}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 0: G5_lowvar_s42  (seed 42 고정 새 기준선)"
    echo "  참고 기준: PSNR 25.664 / SSIM 0.96059 / LPIPS 0.02414 (seed 미고정)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "G5_lowvar_s42" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
        done
    done
else
    echo "  [SKIP] G5_lowvar_s42 (ENABLE_G5_LOWVAR_S42=0)"
fi

# ================================================================
# 세트 1: HM2_s42  (seed 42 고정 HM2 재검증)
#
#   목적: 이전 HM2 결과(+0.055 dB)가 seed 노이즈인지
#         실제 신호인지 seed 42 고정으로 재검증.
#         G5_lowvar_s42 대비 유의미한 차이가 있으면 HM2 확정.
#   변경점: + --seed 42 + --hard-mining-main-loss
# ================================================================
if [ "${ENABLE_HM2_S42}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: HM2_s42  (seed 42 고정, hard-mining-main-loss 추가)"
    echo "  BASE_ARGS + --seed 42 + --hard-mining-main-loss"
    echo "  비교: G5_lowvar_s42 vs HM2_s42"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "HM2_s42" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --hard-mining-main-loss
        done
    done
else
    echo "  [SKIP] HM2_s42 (ENABLE_HM2_S42=0)"
fi

# ================================================================
# 세트 2: div_lowtarget
#
#   목적: target_var를 0.12 → 0.06으로 낮춰 분화 요구를 줄임.
#         현재 0.12 임계점이 과도한 분화 압력을 유발해
#         MSE 학습을 방해하고 있는지 확인.
#         성능이 오르면 → 0.12가 과도했다는 결론.
#         성능이 나빠지면 → 현재 target이 적절하거나 부족.
#   변경점: --target-diversity-var 0.12 → 0.06
# ================================================================
if [ "${ENABLE_DIV_LOWTARGET}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: div_lowtarget"
    echo "  weight=0.05 (유지), target=0.06 (0.12 → 절반)"
    echo "  비교: G5_lowvar_s42 vs div_lowtarget"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "div_lowtarget" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --target-diversity-var 0.06
        done
    done
else
    echo "  [SKIP] div_lowtarget (ENABLE_DIV_LOWTARGET=0)"
fi

# ================================================================
# 세트 3: div_lowweight
#
#   목적: weight만 0.05 → 0.02로 낮추고 target(0.12)은 유지.
#         임계점은 그대로 두되 loss 강도만 줄였을 때의 효과.
#         div_lowtarget과 비교하면 "임계점 조정"과 "강도 조정"
#         중 어느 쪽이 더 효과적인지 구분 가능.
#   변경점: --weight-diversity-loss 0.05 → 0.02
# ================================================================
if [ "${ENABLE_DIV_LOWWEIGHT}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: div_lowweight"
    echo "  weight=0.02 (0.05 → 낮춤), target=0.12 (유지)"
    echo "  비교: G5_lowvar_s42 vs div_lowweight"
    echo "  참고: div_lowtarget과 비교 → 임계점 vs 강도 조정 효과 분리"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "div_lowweight" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.02
        done
    done
else
    echo "  [SKIP] div_lowweight (ENABLE_DIV_LOWWEIGHT=0)"
fi

# ================================================================
# 세트 4: div_stronger
#
#   목적: weight=0.08, target=0.15 로 둘 다 강화.
#         현재 분화가 아직 부족해서 성능이 제한되는지 확인.
#         성능이 오르면 → 더 공격적인 분화가 유리.
#         성능이 나빠지면 → 현재 수준이 이미 충분하거나 과함.
#   변경점: --weight-diversity-loss 0.05 → 0.08
#           --target-diversity-var  0.12 → 0.15
# ================================================================
if [ "${ENABLE_DIV_STRONGER}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 4: div_stronger"
    echo "  weight=0.08 (↑), target=0.15 (↑) — 둘 다 강화"
    echo "  비교: G5_lowvar_s42 vs div_stronger"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "div_stronger" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var 0.15
        done
    done
else
    echo "  [SKIP] div_stronger (ENABLE_DIV_STRONGER=0)"
fi

# ================================================================
# 세트 5: div_tighter
#
#   목적: weight=0.08 (강하게), target=0.06 (일찍 멈춤) 조합.
#         강한 압력으로 빠르게 분화시키되 과도한 분화는 억제.
#         div_stronger(둘 다 강함)와 div_lowtarget(target만 낮음)의
#         중간 가설: 강도와 임계점을 반대 방향으로 조정.
#   변경점: --weight-diversity-loss 0.05 → 0.08
#           --target-diversity-var  0.12 → 0.06
# ================================================================
if [ "${ENABLE_DIV_TIGHTER}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 5: div_tighter"
    echo "  weight=0.08 (↑ 강하게), target=0.06 (↓ 일찍 멈춤)"
    echo "  비교: G5_lowvar_s42 vs div_tighter"
    echo "  참고: div_stronger(둘 다 강함) vs div_tighter(강도↑+임계↓)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "div_tighter" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-diversity-loss 0.08 \
                --target-diversity-var 0.06
        done
    done
else
    echo "  [SKIP] div_tighter (ENABLE_DIV_TIGHTER=0)"
fi

# ================================================================
# 평가 실행
# G5_lowvar 기존 결과는 파일 재사용 (재실행 없이 비교)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags G5_lowvar \
               G5_lowvar_s42 \
               HM2_s42 \
               div_lowtarget \
               div_lowweight \
               div_stronger \
               div_tighter \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_div_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_div_sweep.csv"
fi

# ── 최종 요약 ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  전체 실행 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  성공: ${SUCCESS} / ${TOTAL}"
if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo "  ✅ 모든 케이스 성공"
else
    echo "  ❌ 실패한 케이스 (${#FAILED_CASES[@]}건):"
    for c in "${FAILED_CASES[@]}"; do
        echo "     - ${c}"
    done
fi
echo "================================================================"
