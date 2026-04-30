#!/usr/bin/env bash
# ================================================================
# run_hsm_sweep2.sh  (v1 — 2026-04-30)
# Hard Slice Mining 2차 하이퍼파라미터 탐색
#   인자: --slice-residual-alpha × --hard-mining-start-iter
#
# ── 1차 탐색(run_hsm_sweep.sh) 결과 요약 ──────────────────────
#   - start=500 > start=1000 일관적 (예상과 반대 방향)
#     → iter 500~1000 구간에도 유효한 잔차 분화가 존재함
#   - alpha 낮출수록 FLAIR↑, T1CE↓ (모달리티 간 방향 분기)
#     FLAIR: time_IC(26.689) < B(26.735) < C(26.741) < D(26.746) < E(26.768) < F(26.794)
#     T1CE:  time_IC(25.841) > B(25.799) > C(25.803) ≈ D(25.798), F(25.827) 소폭 회복
#   - current best overall: hsm_F (alpha=0.8, start=500), PSNR +0.046 전체 평균
#
# ── 2차 탐색 목적 ──────────────────────────────────────────────
#   1) start=0이 start=500보다 나쁜지 재확인
#      → start=0에서도 alpha 보정이 있다면 충분히 경쟁력이 있는가?
#   2) alpha를 0.7, 0.6까지 낮추면 T1CE 손해가 커지는가, 교차점이 있는가?
#
# ── 탐색 구조 ──────────────────────────────────────────────────
#   hsm_H  : alpha=0.9, start=0    — start=0 vs hsm_E(start=500), alpha=0.9 고정
#   hsm_I  : alpha=0.8, start=0    — start=0 vs hsm_F(start=500), alpha=0.8 고정
#   hsm_J  : alpha=0.7, start=500  — alpha 계속 낮춤, best start=500 고정
#   hsm_K  : alpha=0.6, start=500  — alpha 극단, T1CE 반전 여부 탐색
#   hsm_L  : alpha=0.7, start=0    — alpha=0.7에서 start_iter 효과
#
# ── 비교 구조 ─────────────────────────────────────────────────
#   hsm_H vs hsm_E(기존): start=0 vs 500 @ alpha=0.9
#   hsm_I vs hsm_F(기존): start=0 vs 500 @ alpha=0.8
#   hsm_J vs hsm_F(기존): alpha=0.7 vs 0.8 @ start=500
#   hsm_K vs hsm_J:       alpha=0.6 vs 0.7 @ start=500
#   hsm_L vs hsm_J:       start=0  vs 500  @ alpha=0.7
#
# ── 예상 시간 ─────────────────────────────────────────────────
#   세트당: 5환자 × 2모달리티 × 260초 ≈ 43분
#   전체 5세트: 약 215분 (3.6시간)
#   평가 포함 전체: 약 3.7시간
#
# 사용법:
#   chmod +x run_hsm_sweep2.sh
#   ./run_hsm_sweep2.sh                     # 모든 세트 실행 후 평가
#   ENABLE_HSM_H=0 ./run_hsm_sweep2.sh      # HSM_H 건너뜀
#   ENABLE_HSM_I=0 ./run_hsm_sweep2.sh      # HSM_I 건너뜀
#   ENABLE_HSM_J=0 ./run_hsm_sweep2.sh      # HSM_J 건너뜀
#   ENABLE_HSM_K=0 ./run_hsm_sweep2.sh      # HSM_K 건너뜀
#   ENABLE_HSM_L=0 ./run_hsm_sweep2.sh      # HSM_L 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
ENABLE_HSM_H=${ENABLE_HSM_H:-1}
ENABLE_HSM_I=${ENABLE_HSM_I:-1}
ENABLE_HSM_J=${ENABLE_HSM_J:-1}
ENABLE_HSM_K=${ENABLE_HSM_K:-1}
ENABLE_HSM_L=${ENABLE_HSM_L:-1}

# ── BASE_ARGS: time_IC 완전 재현 ───────────────────────────────
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
  --weight-ff-loss 0.05
  --ff-alpha 2.0
  --n-iter 2000
  --batch-size 4096
  --learning-rate 5e-3
  --gamma 0.33
  --milestones 0.5 0.75 0.9
  --seed 42
"

INTENSITY_MEAN=308
DEGRADED_ROOT="/data/BraTS20_Degraded_4x_5"

PATIENTS=(003 026 030 040 060)
MODALITIES=(flair t1ce)

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
# 세트 HSM_H: alpha=0.9, start_iter=0
#
#   목적: start=0에서 alpha=0.9 보정이 있을 때의 성능 확인.
#         1차에서 start=500 > start=1000이 확인됐으므로,
#         start=0이 그보다 더 나쁜지, 아니면 alpha 보정으로
#         충분히 경쟁력을 가지는지 검증.
#   비교: hsm_E(alpha=0.9, start=500) vs HSM_H → start_iter 효과 @ alpha=0.9
#   예측: HSM_H < hsm_E  (start=0 초기 노이즈 구간이 alpha=0.9로도 극복 안 됨)
#   EMA 잔존율: alpha=0.9, k=20 → 12.3% (노이즈 구간 포함 업데이트)
# ================================================================
if [ "${ENABLE_HSM_H}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_H: alpha=0.9, start_iter=0"
    echo "  비교: hsm_E(start=500) vs HSM_H → start=0 vs 500 @ alpha=0.9"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_H" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.9 \
                --hard-mining-start-iter 0
        done
    done
else
    echo "  [SKIP] HSM_H (ENABLE_HSM_H=0)"
fi

# ================================================================
# 세트 HSM_I: alpha=0.8, start_iter=0
#
#   목적: start=0에서 alpha=0.8의 성능 확인.
#         alpha=0.8은 k=20 기준 잔존율 1.2%로 사실상 완전 수렴.
#         start=0이어도 초기 노이즈를 EMA가 빠르게 덮어쓰므로
#         start=500과 차이가 없을 수도 있음.
#   비교: hsm_F(alpha=0.8, start=500) vs HSM_I → start_iter 효과 @ alpha=0.8
#   예측: HSM_I ≈ hsm_F  (alpha 낮을수록 start_iter 영향 줄어들 것)
#   EMA 잔존율: alpha=0.8, k=20 → 1.2%
# ================================================================
if [ "${ENABLE_HSM_I}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_I: alpha=0.8, start_iter=0"
    echo "  비교: hsm_F(start=500) vs HSM_I → start=0 vs 500 @ alpha=0.8"
    echo "  예측: alpha 낮을수록 start_iter 영향 소멸 가능"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_I" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.8 \
                --hard-mining-start-iter 0
        done
    done
else
    echo "  [SKIP] HSM_I (ENABLE_HSM_I=0)"
fi

# ================================================================
# 세트 HSM_J: alpha=0.7, start_iter=500
#
#   목적: alpha를 0.8→0.7로 낮췄을 때 T1CE 패턴 확인.
#         1차에서 FLAIR는 alpha 낮을수록 일관 상승,
#         T1CE는 alpha 낮을수록 미약하게 하락하는 패턴이 있었음.
#         0.7까지 낮추면 T1CE 손해가 더 커지는가, 아니면 교차점이 있는가?
#   비교: hsm_F(alpha=0.8, start=500) vs HSM_J → alpha=0.7 추가 하락 여부
#   EMA 잔존율: alpha=0.7, k=20 → 약 0.8%
# ================================================================
if [ "${ENABLE_HSM_J}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_J: alpha=0.7, start_iter=500"
    echo "  비교: hsm_F(alpha=0.8) vs HSM_J → alpha 추가 하락 효과"
    echo "  관심: T1CE 손해 누적 vs FLAIR 개선 지속 여부"
    echo "  EMA 잔존율: k=20 기준 ~0.8%"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_J" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.7 \
                --hard-mining-start-iter 500
        done
    done
else
    echo "  [SKIP] HSM_J (ENABLE_HSM_J=0)"
fi

# ================================================================
# 세트 HSM_K: alpha=0.6, start_iter=500
#
#   목적: alpha 극단값 탐색.
#         alpha=0.6이면 k=20에서 잔존율 ~0.04% — EMA가 현재 배치에만 반응.
#         사실상 sliding window 최솟값에 가까운 동작.
#         FLAIR에서 추가 개선이 멈추거나, T1CE 손해가 급격히 커지는
#         임계점을 확인하기 위한 경계 탐색.
#   비교: HSM_K vs HSM_J → alpha=0.6 vs 0.7 @ start=500
#   EMA 잔존율: alpha=0.6, k=20 → 약 0.04% (사실상 0)
# ================================================================
if [ "${ENABLE_HSM_K}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_K: alpha=0.6, start_iter=500"
    echo "  비교: HSM_J(alpha=0.7) vs HSM_K → alpha 극단값 경계 탐색"
    echo "  주의: EMA가 현재 배치 평균에만 반응 — 진동/불안정 가능성"
    echo "  EMA 잔존율: k=20 기준 ~0.04%"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_K" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.6 \
                --hard-mining-start-iter 500
        done
    done
else
    echo "  [SKIP] HSM_K (ENABLE_HSM_K=0)"
fi

# ================================================================
# 세트 HSM_L: alpha=0.7, start_iter=0
#
#   목적: alpha=0.7에서 start_iter의 영향을 격리.
#         HSM_I(alpha=0.8, start=0) ≈ hsm_F(alpha=0.8, start=500) 라면,
#         alpha가 낮아질수록 start_iter 효과가 소멸한다는 가설을 검증.
#         HSM_L vs HSM_J 비교로 alpha=0.7에서도 동일하게 성립하는지 확인.
#   비교: HSM_J(alpha=0.7, start=500) vs HSM_L → start_iter 효과 @ alpha=0.7
#   EMA 잔존율: alpha=0.7, k=20 → ~0.8%
# ================================================================
if [ "${ENABLE_HSM_L}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_L: alpha=0.7, start_iter=0"
    echo "  비교: HSM_J(start=500) vs HSM_L → start=0 vs 500 @ alpha=0.7"
    echo "  가설: alpha 낮을수록 start_iter 효과 소멸 → HSM_L ≈ HSM_J"
    echo "  EMA 잔존율: k=20 기준 ~0.8%"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_L" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.7 \
                --hard-mining-start-iter 0
        done
    done
else
    echo "  [SKIP] HSM_L (ENABLE_HSM_L=0)"
fi

# ================================================================
# 평가 실행
#   1차 결과(time_IC, hsm_B~G)와 함께 전체 비교
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  2차 신규 5세트: hsm_H / hsm_I / hsm_J / hsm_K / hsm_L"
echo "  1차 주요 기준점 포함: time_IC / hsm_E / hsm_F"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags time_IC \
               hsm_E \
               hsm_F \
               hsm_H \
               hsm_I \
               hsm_J \
               hsm_K \
               hsm_L \
    --patients 003 026 030 040 060 \
    --modalities flair t1ce \
    --output-csv eval_hsm_sweep2.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_hsm_sweep2.csv"
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
