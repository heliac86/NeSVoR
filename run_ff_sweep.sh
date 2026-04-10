#!/usr/bin/env bash
# ================================================================
# run_ff_sweep.sh  (v1 — 2026-04-10)
# FF Loss 하이퍼파라미터 스윕
# (--weight-ff-loss × --ff-alpha 교호작용 탐색)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# 현재까지 확정된 누적 개선:
#   G5_lowvar_s42     : PSNR 25.496  ← seed42 기준선
#   div_tighter       : PSNR 25.567  (+0.071)
#   div_tighter_HM2   : PSNR 25.627  (+0.131)  ← 현재 BASE
#
# FF loss 구조:
#   L_total += weight_ff_loss × Σ W(f) · |FFT(pred) - FFT(gt)|²
#   W(f) = |FFT(pred) - FFT(gt)|^alpha  (dynamic freq weighting)
#
#   alpha=0   → 모든 주파수 동등 가중치
#   alpha↑    → 현재 가장 어려운 주파수에 집중
#               실질적 loss 크기 증가 → weight를 낮춰야 안정
#   weight↑   → FF loss 전체 스케일 증가
#               alpha 낮을 때는 안정, alpha 높을 때는 발산 위험
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q0. FF loss를 완전히 제거하면 어떻게 되는가?        → ff_w000
# Q1. weight를 줄이면 개선되는가?                    → ff_w005_a10
# Q2. weight를 올리면 개선되는가?                    → ff_w015_a10
# Q3. weight 상한은 어디인가?                        → ff_w020_a10
# Q4. alpha를 완만하게 하면 어떻게 되는가?            → ff_w010_a05
# Q5. alpha를 강하게 하면 어떻게 되는가?              → ff_w010_a20
# Q6. weight 낮음 + alpha 강함 조합은?               → ff_w005_a20
# Q7. weight 높음 + alpha 완만 조합은?               → ff_w015_a05
#
# ── 실험 세트 ──────────────────────────────────────────────────
#   [0] ff_w000       : weight=0.0,  alpha=—    (FF 완전 제거 ablation)
#   [1] ff_w005_a10   : weight=0.05, alpha=1.0  (weight 축 저점)
#   [2] ff_w015_a10   : weight=0.15, alpha=1.0  (weight 축 중상단)
#   [3] ff_w020_a10   : weight=0.20, alpha=1.0  (weight 축 상한)
#   [4] ff_w010_a05   : weight=0.10, alpha=0.5  (alpha 축 완만)
#   [5] ff_w010_a20   : weight=0.10, alpha=2.0  (alpha 축 강함)
#   [6] ff_w005_a20   : weight=0.05, alpha=2.0  (교호작용 핵심 후보)
#   [7] ff_w015_a05   : weight=0.15, alpha=0.5  (반대 극단)
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 8세트 x 15 = 120
#
# 사용법:
#   chmod +x run_ff_sweep.sh
#   ./run_ff_sweep.sh                          # 모든 세트 실행
#   ENABLE_FF_W000=0     ./run_ff_sweep.sh     # ff_w000 건너뜀
#   ENABLE_FF_W005_A10=0 ./run_ff_sweep.sh     # ff_w005_a10 건너뜀
#   ENABLE_FF_W015_A10=0 ./run_ff_sweep.sh     # ff_w015_a10 건너뜀
#   ENABLE_FF_W020_A10=0 ./run_ff_sweep.sh     # ff_w020_a10 건너뜀
#   ENABLE_FF_W010_A05=0 ./run_ff_sweep.sh     # ff_w010_a05 건너뜀
#   ENABLE_FF_W010_A20=0 ./run_ff_sweep.sh     # ff_w010_a20 건너뜀
#   ENABLE_FF_W005_A20=0 ./run_ff_sweep.sh     # ff_w005_a20 건너뜀
#   ENABLE_FF_W015_A05=0 ./run_ff_sweep.sh     # ff_w015_a05 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_FF_W000=${ENABLE_FF_W000:-1}
ENABLE_FF_W005_A10=${ENABLE_FF_W005_A10:-1}
ENABLE_FF_W015_A10=${ENABLE_FF_W015_A10:-1}
ENABLE_FF_W020_A10=${ENABLE_FF_W020_A10:-1}
ENABLE_FF_W010_A05=${ENABLE_FF_W010_A05:-1}
ENABLE_FF_W010_A20=${ENABLE_FF_W010_A20:-1}
ENABLE_FF_W005_A20=${ENABLE_FF_W005_A20:-1}
ENABLE_FF_W015_A05=${ENABLE_FF_W015_A05:-1}

# ── BASE_ARGS: div_tighter_HM2 완전 재현 인자 ─────────────────
# --weight-ff-loss, --ff-alpha 는 세트별로 명시적으로 덮어씀
BASE_ARGS="
  --output-resolution 1.0
  --no-transformation-optimization
  --registration none
  --single-precision
  --patch-size 48
  --n-patches 8
  --weight-image 1.0
  --delta 0.05
  --weight-diversity-loss 0.08
  --target-diversity-var 0.06
  --diversity-loss-fn variance
  --diversity-loss-space raw
  --gating-grad-clip 0.0
  --hard-mining-main-loss
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
# 세트 0: ff_w000  (FF loss 완전 제거 ablation)
#
#   목적: FF loss를 아예 없앴을 때 성능이 얼마나 떨어지는가.
#         FF loss의 기여 자체를 정량화하는 역방향 기준점.
#         성능이 크게 떨어지면 → FF loss가 핵심 구성요소.
#         성능 차이가 작으면 → FF loss의 실질 기여 재검토 필요.
#   변경점: --weight-ff-loss 0.0 (--ff-alpha 생략)
# ================================================================
if [ "${ENABLE_FF_W000}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 0: ff_w000  (FF loss 완전 제거 ablation)"
    echo "  weight=0.0 (FF 제거)"
    echo "  비교: div_tighter_HM2 (PSNR 25.627) vs ff_w000"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w000" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.0
        done
    done
else
    echo "  [SKIP] ff_w000 (ENABLE_FF_W000=0)"
fi

# ================================================================
# 세트 1: ff_w005_a10  (weight 축 저점)
#
#   목적: weight를 0.10 → 0.05로 절반 줄임. alpha 유지.
#         FF loss 기여를 줄였을 때 성능 변화 확인.
#         성능이 오르면 → 현재 0.10이 과도.
#         성능이 나빠지면 → 0.10이 적절하거나 부족.
#   변경점: --weight-ff-loss 0.10 → 0.05
# ================================================================
if [ "${ENABLE_FF_W005_A10}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 1: ff_w005_a10  (weight 축 저점)"
    echo "  weight=0.05 (↓절반), alpha=1.0 (유지)"
    echo "  비교: div_tighter_HM2 vs ff_w005_a10"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w005_a10" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.05 \
                --ff-alpha 1.0
        done
    done
else
    echo "  [SKIP] ff_w005_a10 (ENABLE_FF_W005_A10=0)"
fi

# ================================================================
# 세트 2: ff_w015_a10  (weight 축 중상단)
#
#   목적: weight를 0.10 → 0.15로 올림. alpha 유지.
#         FF loss를 강화했을 때 성능 변화 확인.
#   변경점: --weight-ff-loss 0.10 → 0.15
# ================================================================
if [ "${ENABLE_FF_W015_A10}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 2: ff_w015_a10  (weight 축 중상단)"
    echo "  weight=0.15 (↑1.5배), alpha=1.0 (유지)"
    echo "  비교: div_tighter_HM2 vs ff_w015_a10"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w015_a10" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.15 \
                --ff-alpha 1.0
        done
    done
else
    echo "  [SKIP] ff_w015_a10 (ENABLE_FF_W015_A10=0)"
fi

# ================================================================
# 세트 3: ff_w020_a10  (weight 축 상한 탐색)
#
#   목적: weight=0.20으로 상한 탐색.
#         FF loss가 MSE를 압도하기 시작하는 임계점 확인.
#         성능이 급격히 나빠지면 → 0.20은 상한 초과.
#   변경점: --weight-ff-loss 0.10 → 0.20
# ================================================================
if [ "${ENABLE_FF_W020_A10}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 3: ff_w020_a10  (weight 축 상한)"
    echo "  weight=0.20 (↑2배), alpha=1.0 (유지)"
    echo "  비교: div_tighter_HM2 vs ff_w020_a10"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w020_a10" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.20 \
                --ff-alpha 1.0
        done
    done
else
    echo "  [SKIP] ff_w020_a10 (ENABLE_FF_W020_A10=0)"
fi

# ================================================================
# 세트 4: ff_w010_a05  (alpha 축 완만)
#
#   목적: alpha를 1.0 → 0.5로 낮춰 주파수 가중치를 완만하게.
#         모든 주파수를 더 균등하게 학습할 때의 효과 확인.
#         weight는 0.10 유지.
#   변경점: --ff-alpha 1.0 → 0.5
# ================================================================
if [ "${ENABLE_FF_W010_A05}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 4: ff_w010_a05  (alpha 축 완만)"
    echo "  weight=0.10 (유지), alpha=0.5 (↓완만)"
    echo "  비교: div_tighter_HM2 vs ff_w010_a05"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w010_a05" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.10 \
                --ff-alpha 0.5
        done
    done
else
    echo "  [SKIP] ff_w010_a05 (ENABLE_FF_W010_A05=0)"
fi

# ================================================================
# 세트 5: ff_w010_a20  (alpha 축 강함)
#
#   목적: alpha를 1.0 → 2.0으로 올려 어려운 주파수에 집중.
#         고주파 성분(엣지, 세부 구조)에 더 많은 학습 압력.
#         weight는 0.10 유지 — alpha 상승으로 실질 loss 증가 주의.
#   변경점: --ff-alpha 1.0 → 2.0
# ================================================================
if [ "${ENABLE_FF_W010_A20}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 5: ff_w010_a20  (alpha 축 강함)"
    echo "  weight=0.10 (유지), alpha=2.0 (↑강함)"
    echo "  비교: div_tighter_HM2 vs ff_w010_a20"
    echo "  주의: alpha↑ → 실질 loss 증가 → 발산 가능성 모니터링"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w010_a20" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.10 \
                --ff-alpha 2.0
        done
    done
else
    echo "  [SKIP] ff_w010_a20 (ENABLE_FF_W010_A20=0)"
fi

# ================================================================
# 세트 6: ff_w005_a20  (교호작용 핵심 후보)
#
#   목적: weight 낮춤 + alpha 강함 조합.
#         alpha↑로 인한 실질 loss 증가를 weight↓로 상쇄.
#         "어려운 주파수에 집중하되 전체 강도는 유지" 가설.
#         ff_w010_a20(alpha만 올림)과 비교하면 weight 보정 효과 분리.
#   변경점: --weight-ff-loss 0.10 → 0.05 / --ff-alpha 1.0 → 2.0
# ================================================================
if [ "${ENABLE_FF_W005_A20}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 6: ff_w005_a20  (교호작용 핵심 후보)"
    echo "  weight=0.05 (↓), alpha=2.0 (↑) — 강도 상쇄 조합"
    echo "  비교: div_tighter_HM2 vs ff_w005_a20"
    echo "  참고: ff_w010_a20과 비교 → weight 보정 효과 분리"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w005_a20" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.05 \
                --ff-alpha 2.0
        done
    done
else
    echo "  [SKIP] ff_w005_a20 (ENABLE_FF_W005_A20=0)"
fi

# ================================================================
# 세트 7: ff_w015_a05  (반대 극단)
#
#   목적: weight 높음 + alpha 완만 조합.
#         주파수 가중치는 고르게 유지하면서 전체 FF 기여를 높임.
#         ff_w005_a20(반대 방향)과 비교하면 어느 전략이 우월한지 판단.
#   변경점: --weight-ff-loss 0.10 → 0.15 / --ff-alpha 1.0 → 0.5
# ================================================================
if [ "${ENABLE_FF_W015_A05}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 7: ff_w015_a05  (반대 극단)"
    echo "  weight=0.15 (↑), alpha=0.5 (↓) — 균등 강화 조합"
    echo "  비교: div_tighter_HM2 vs ff_w015_a05"
    echo "  참고: ff_w005_a20(반대 극단)과 전략 비교"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "ff_w015_a05" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --weight-ff-loss 0.15 \
                --ff-alpha 0.5
        done
    done
else
    echo "  [SKIP] ff_w015_a05 (ENABLE_FF_W015_A05=0)"
fi

# ================================================================
# 평가 실행
# div_tighter_HM2 기존 결과는 파일 재사용 (재실행 없음)
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags div_tighter_HM2 \
               ff_w000 \
               ff_w005_a10 \
               ff_w015_a10 \
               ff_w020_a10 \
               ff_w010_a05 \
               ff_w010_a20 \
               ff_w005_a20 \
               ff_w015_a05 \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_ff_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_ff_sweep.csv"
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
