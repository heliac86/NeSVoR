#!/usr/bin/env bash
# ================================================================
# run_hsm_sweep.sh  (v1 — 2026-04-29)
# Hard Slice Mining 하이퍼파라미터 탐색
#   인자: --slice-residual-alpha × --hard-mining-start-iter
#
# ── 탐색 구조 ──────────────────────────────────────────────────
#   HSM_A  : baseline (alpha=0.99, start=0)    — 재실행 없음, time_IC 재사용
#   HSM_B  : alpha=0.99, start=1000            — start_iter 단독 교정
#   HSM_C  : alpha=0.9,  start=1000            — 1순위 후보
#   HSM_D  : alpha=0.8,  start=1000            — 2순위 후보
#   HSM_E  : alpha=0.9,  start=500             — start_iter 조기 vs HSM_C
#   HSM_F  : alpha=0.8,  start=500             — 조기 가동 + 빠른 수렴
#   HSM_G  : alpha=0.85, start=1000            — C와 D 사이 보간
#
# ── 진단 논거 ──────────────────────────────────────────────────
#   k=19.9회 실측 기반 초기값 잔존율:
#     alpha=0.99 → 81.9%  (EMA 거의 노이즈)
#     alpha=0.9  → 12.3%  (수렴 적절)
#     alpha=0.85 →  5.0%  (C-D 보간)
#     alpha=0.8  →  1.2%  (공격적 수렴)
#   start_iter=0   → iter 1~1000 노이즈 잔차가 EMA에 누적됨
#   start_iter=500 → iter 1~500 노이즈 구간 포함
#   start_iter=1000 → 첫 LR decay(milestone 0.5) 직후 HSM 가동
#
# ── 비교 설계 ──────────────────────────────────────────────────
#   HSM_B vs HSM_A : start_iter 단독 효과
#   HSM_C vs HSM_B : alpha 단독 효과 (start=1000 고정)
#   HSM_D vs HSM_C : alpha 강화 효과 (0.9→0.8)
#   HSM_G vs HSM_C/D : alpha 보간 (0.85)
#   HSM_E vs HSM_C : start_iter 조기화 효과 (alpha=0.9 고정)
#   HSM_F vs HSM_D : start_iter 조기화 효과 (alpha=0.8 고정)
#
# ── 예상 시간 ─────────────────────────────────────────────────
#   세트당: 5환자 × 2모달리티 × 260초 ≈ 43분
#   전체 6세트 신규: 약 260분 (4.3시간)
#   평가 포함 전체: 약 4.5시간
#
# 사용법:
#   chmod +x run_hsm_sweep.sh
#   ./run_hsm_sweep.sh                     # 모든 신규 세트 실행 후 평가
#   ENABLE_HSM_B=0 ./run_hsm_sweep.sh      # HSM_B 건너뜀
#   ENABLE_HSM_C=0 ./run_hsm_sweep.sh      # HSM_C 건너뜀
#   ENABLE_HSM_D=0 ./run_hsm_sweep.sh      # HSM_D 건너뜀
#   ENABLE_HSM_E=0 ./run_hsm_sweep.sh      # HSM_E 건너뜀
#   ENABLE_HSM_F=0 ./run_hsm_sweep.sh      # HSM_F 건너뜀
#   ENABLE_HSM_G=0 ./run_hsm_sweep.sh      # HSM_G 건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
# HSM_A (time_IC baseline)는 재실행하지 않고 기존 결과 재사용
ENABLE_HSM_B=${ENABLE_HSM_B:-1}
ENABLE_HSM_C=${ENABLE_HSM_C:-1}
ENABLE_HSM_D=${ENABLE_HSM_D:-1}
ENABLE_HSM_E=${ENABLE_HSM_E:-1}
ENABLE_HSM_F=${ENABLE_HSM_F:-1}
ENABLE_HSM_G=${ENABLE_HSM_G:-1}

# ── BASE_ARGS: time_IC 완전 재현 인자 ─────────────────────────
# --slice-residual-alpha, --hard-mining-start-iter 는 세트별로 명시적으로 덮어씀
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
# 세트 HSM_B: alpha=0.99, start_iter=1000
#
#   목적: start_iter 단독 교정 효과 확인.
#         alpha는 baseline(0.99)과 동일하게 유지.
#         HSM_B > HSM_A  → start_iter 자체가 독립 효과 있음
#         HSM_B ≈ HSM_A  → alpha 교정이 핵심임을 방증 → C/D 기대 상승
#   변경: --hard-mining-start-iter 1000
#   EMA 잔존율: alpha=0.99, k≈10(start 이후) → 잔존율 ~90%
# ================================================================
if [ "${ENABLE_HSM_B}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_B: alpha=0.99, start_iter=1000"
    echo "  비교: HSM_A(time_IC, start=0) vs HSM_B → start_iter 단독 효과"
    echo "  EMA 잔존율: ~90% (start 이후 k≈10회 기준)"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_B" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.99 \
                --hard-mining-start-iter 1000
        done
    done
else
    echo "  [SKIP] HSM_B (ENABLE_HSM_B=0)"
fi

# ================================================================
# 세트 HSM_C: alpha=0.9, start_iter=1000  [1순위 후보]
#
#   목적: 수학적으로 가장 균형잡힌 조합.
#         start_iter=1000으로 노이즈 구간 배제,
#         alpha=0.9로 초기값 잔존율을 81.9%→12.3%로 개선.
#         HSM_C > HSM_B → alpha 교정의 독립 효과 확인
#         HSM_C > HSM_A → 두 인자 교정의 복합 효과 확인
#   변경: --slice-residual-alpha 0.9  --hard-mining-start-iter 1000
#   EMA 잔존율: alpha=0.9, k≈10(start 이후) → 잔존율 ~35%
#              alpha=0.9, k=20(전체) → 잔존율 12.3%
# ================================================================
if [ "${ENABLE_HSM_C}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_C: alpha=0.9, start_iter=1000  [1순위 후보]"
    echo "  비교: HSM_B(alpha=0.99) vs HSM_C → alpha 단독 효과"
    echo "  EMA 잔존율: 전체 k=20 기준 12.3%  (HSM_A 대비 81.9%→12.3%)"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_C" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.9 \
                --hard-mining-start-iter 1000
        done
    done
else
    echo "  [SKIP] HSM_C (ENABLE_HSM_C=0)"
fi

# ================================================================
# 세트 HSM_D: alpha=0.8, start_iter=1000  [2순위 후보]
#
#   목적: 더 공격적인 수렴. 최근 배치 잔차에 민감하게 반응.
#         HSM_D > HSM_C → 더 빠른 수렴이 유리한 환경
#         HSM_D < HSM_C → alpha=0.9가 최적, 0.8은 단기 노이즈 과민
#         HSM_D ≈ HSM_C → 0.8~0.9 사이 둔감 → HSM_G로 보간 불필요
#   변경: --slice-residual-alpha 0.8  --hard-mining-start-iter 1000
#   EMA 잔존율: alpha=0.8, k=20 → 1.2%  (사실상 완전 수렴)
# ================================================================
if [ "${ENABLE_HSM_D}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_D: alpha=0.8, start_iter=1000  [2순위 후보]"
    echo "  비교: HSM_C(alpha=0.9) vs HSM_D → alpha 강화 효과"
    echo "  EMA 잔존율: 전체 k=20 기준 1.2%  (사실상 완전 수렴)"
    echo "  주의: 단기 배치 노이즈 과민 가능성 — FLAIR/T1CE 방향 일치 여부 확인"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_D" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.8 \
                --hard-mining-start-iter 1000
        done
    done
else
    echo "  [SKIP] HSM_D (ENABLE_HSM_D=0)"
fi

# ================================================================
# 세트 HSM_E: alpha=0.9, start_iter=500
#
#   목적: start_iter 조기화의 영향 격리 (alpha=0.9 고정).
#         HSM_E ≈ HSM_C → start_iter 500/1000 차이 무의미
#         HSM_E < HSM_C → start_iter=1000이 유효함 재확인
#         HSM_E > HSM_C → iter 500~1000 구간에도 유효한 잔차 분화 있음
#   변경: --slice-residual-alpha 0.9  --hard-mining-start-iter 500
# ================================================================
if [ "${ENABLE_HSM_E}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_E: alpha=0.9, start_iter=500"
    echo "  비교: HSM_C(start=1000) vs HSM_E → start_iter 조기화 효과 (alpha=0.9 고정)"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_E" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.9 \
                --hard-mining-start-iter 500
        done
    done
else
    echo "  [SKIP] HSM_E (ENABLE_HSM_E=0)"
fi

# ================================================================
# 세트 HSM_F: alpha=0.8, start_iter=500
#
#   목적: 조기 가동 + 공격적 수렴 조합.
#         HSM_F vs HSM_D → alpha=0.8에서 start_iter 조기화 효과
#         HSM_F vs HSM_E → start_iter=500에서 alpha 강화 효과
#   변경: --slice-residual-alpha 0.8  --hard-mining-start-iter 500
# ================================================================
if [ "${ENABLE_HSM_F}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_F: alpha=0.8, start_iter=500"
    echo "  비교: HSM_D(start=1000) vs HSM_F → alpha=0.8에서 start_iter 조기화"
    echo "  비교: HSM_E(alpha=0.9)  vs HSM_F → start=500에서 alpha 강화"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_F" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.8 \
                --hard-mining-start-iter 500
        done
    done
else
    echo "  [SKIP] HSM_F (ENABLE_HSM_F=0)"
fi

# ================================================================
# 세트 HSM_G: alpha=0.85, start_iter=1000
#
#   목적: HSM_C(0.9)와 HSM_D(0.8) 사이 보간.
#         C > D이면 → 0.85가 최적일 가능성 탐색
#         D ≥ C이면 → 0.85는 불필요 (하지만 선행 실험 없이 같이 실행)
#   EMA 잔존율: alpha=0.85, k=20 → 약 5.0%
#   변경: --slice-residual-alpha 0.85  --hard-mining-start-iter 1000
# ================================================================
if [ "${ENABLE_HSM_G}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 HSM_G: alpha=0.85, start_iter=1000  [C-D 보간]"
    echo "  비교: HSM_C(0.9) vs HSM_G(0.85) vs HSM_D(0.8) → alpha 세밀 탐색"
    echo "  EMA 잔존율: k=20 기준 ~5.0%"
    echo "  예상 시간: ~260초/케이스, 총 ~43분"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "hsm_G" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --slice-residual-alpha 0.85 \
                --hard-mining-start-iter 1000
        done
    done
else
    echo "  [SKIP] HSM_G (ENABLE_HSM_G=0)"
fi

# ================================================================
# 평가 실행
#   HSM_A = time_IC 기존 결과 재사용 (재실행 없음)
#   총 7세트 비교
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  총 7세트: time_IC(HSM_A) / hsm_B / hsm_C / hsm_D / hsm_E / hsm_F / hsm_G"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags time_IC \
               hsm_B \
               hsm_C \
               hsm_D \
               hsm_E \
               hsm_F \
               hsm_G \
    --patients 003 026 030 040 060 \
    --modalities flair t1ce \
    --output-csv eval_hsm_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_hsm_sweep.csv"
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
