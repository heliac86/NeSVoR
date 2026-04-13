#!/usr/bin/env bash
# ================================================================
# run_time_sweep.sh  (v2 — 2026-04-13)
# 학습 시간 단축 하이퍼파라미터 스윕
# (--n-iter 단독 단축 축 / --batch-size × --learning-rate 축)
#
# ── 실험 배경 ──────────────────────────────────────────────────
# 현재까지 확정된 누적 개선:
#   G5_lowvar_s42     : PSNR 25.496  ← seed42 기준선
#   div_tighter       : PSNR 25.567  (+0.071)
#   div_tighter_HM2   : PSNR 25.627  (+0.131)
#   ff_w005_a20       : PSNR 25.726  (+0.230)  ← 현재 BASE (기존 결과 재사용)
#
# 현재 기본 학습 설정:
#   --n-iter      6000
#   --batch-size  4096
#   --learning-rate 5e-3
#   --milestones  0.5 0.75 0.9  (n_iter 대비 비율 → 자동 추종)
#   케이스당 학습 시간: ~800초
#
# ── 탐색 축 ────────────────────────────────────────────────────
# [축 1] n_iter 단독 단축 (패치 비용 포함 모든 비용이 비례 감소)
#   I-A : n_iter=4000  batch=4096  lr=5e-3       → 예상 단축 ~33%
#   I-B : n_iter=3000  batch=4096  lr=5e-3       → 예상 단축 ~50%
#   I-C : n_iter=2000  batch=4096  lr=5e-3       → 예상 단축 ~67%
#   ※ I-A/I-B/I-C는 v1에서 완료 → 기본값 SKIP (평가에만 참여)
#
# [축 2] batch_size 증가 + n_iter 비례 감소 (Linear Scaling Rule)
#   B-A : n_iter=3000  batch=8192  lr=1e-2       → 2x batch + linear lr
#   B-B : n_iter=3000  batch=8192  lr=5e-3       → 2x batch + lr 고정 (비교군)
#
# [축 3] 수렴 전략 / lr 단독 효과 보완 탐색
#   Z   : n_iter=3000  batch=4096  lr=5e-3  gamma=0.1
#         I-B와 n_iter/batch/lr 동일, gamma만 변경
#         → lr decay 강도 단독 효과 분리
#         → 짧은 학습에서 강한 decay가 수렴에 유리한지 확인
#   Y   : n_iter=3000  batch=4096  lr=8e-3
#         I-B(lr=5e-3), B-B(batch=8192,lr=5e-3), B-A(batch=8192,lr=1e-2)와
#         함께 2×2 factorial 완성 → lr 단독 효과 분리
#         lr=8e-3은 B-A(lr=1e-2)와 I-B(lr=5e-3) 사이 중간점
#   X   : n_iter=5000  batch=4096  lr=5e-3
#         I-A(4000)와 BASE(6000) 사이 보간
#         → I-A에서 손실이 유의미할 경우 임계점 구간 좁히기
#
# ── 2×2 Factorial 구조 (n_iter=3000 기준) ─────────────────────
#              lr=5e-3        lr↑(8e-3/1e-2)
#   batch=4096   I-B            Y (lr=8e-3)
#   batch=8192   B-B            B-A (lr=1e-2)
#   → lr과 batch의 상호작용 분리 가능
#
# ── 평가 시 BASE ───────────────────────────────────────────────
#   ff_w005_a20 : 기존 결과 파일 재사용 (재실행 없음)
#   I-A/I-B/I-C : v1 완료 결과 파일 재사용 (재실행 없음)
#   평가 시 총 9세트 비교: ff_w005_a20 + I-A/I-B/I-C + B-A/B-B + Z/Y/X
#
# ── 핵심 질문 ──────────────────────────────────────────────────
# Q1. n_iter만 줄여도 성능 손실이 허용 범위인가?          → I-A, I-B, I-C
# Q2. n_iter 단축의 성능 손실 임계점은 어디인가?          → I-A vs I-B vs I-C vs X
# Q3. batch 증가 + n_iter 감소 조합이 추가 단축을 주는가?  → B-A, B-B vs I-B
# Q4. lr linear scaling이 batch 조합에서 유효한가?        → B-A vs B-B
# Q5. lr 단독 증가의 효과는?                             → Y vs I-B
# Q6. lr decay 강화가 짧은 학습에 유리한가?               → Z vs I-B
# Q7. I-A~BASE 사이 임계점 구간은?                       → X vs I-A vs BASE
#
# 대상 환자  : 003 026 030 040 060
# 모달리티   : flair / t2 / t1ce
# 총 케이스  : 신규 5세트(B-A/B-B/Z/Y/X) × 15 = 75  (v2 신규 실행분)
#             평가 참여 전체: 9세트 × 15 = 135
#
# 사용법:
#   chmod +x run_time_sweep.sh
#   ./run_time_sweep.sh                    # v2 신규 세트만 실행 후 전체 평가
#   ENABLE_IA=1 ./run_time_sweep.sh        # I-A 재실행 포함
#   ENABLE_IB=1 ./run_time_sweep.sh        # I-B 재실행 포함
#   ENABLE_IC=1 ./run_time_sweep.sh        # I-C 재실행 포함
#   ENABLE_BA=0 ./run_time_sweep.sh        # B-A 건너뜀
#   ENABLE_BB=0 ./run_time_sweep.sh        # B-B 건너뜀
#   ENABLE_Z=0  ./run_time_sweep.sh        # Z   건너뜀
#   ENABLE_Y=0  ./run_time_sweep.sh        # Y   건너뜀
#   ENABLE_X=0  ./run_time_sweep.sh        # X   건너뜀
#
# 출력 파일 명명 규칙:
#   {PATIENT}_{MODALITY}_4x5_{EXP_TAG}.nii.gz
#   {PATIENT}_{MODALITY}_{EXP_TAG}_model.pt
# ================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── 세트별 활성화 플래그 ─────────────────────────────────────
# I-A/I-B/I-C: v1에서 완료 → 기본값 0 (평가에만 참여)
ENABLE_IA=${ENABLE_IA:-0}
ENABLE_IB=${ENABLE_IB:-0}
ENABLE_IC=${ENABLE_IC:-0}
# B-A/B-B: v2 신규 실행
ENABLE_BA=${ENABLE_BA:-1}
ENABLE_BB=${ENABLE_BB:-1}
# X/Y/Z: v2 신규 실행
ENABLE_Z=${ENABLE_Z:-1}
ENABLE_Y=${ENABLE_Y:-1}
ENABLE_X=${ENABLE_X:-1}

# ── BASE_ARGS: ff_w005_a20 완전 재현 인자 ─────────────────────
# --n-iter, --batch-size, --learning-rate, --gamma 는 세트별로 명시적으로 덮어씀
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
  --milestones 0.5 0.75 0.9
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
# 세트 I-A: n_iter=4000  (n_iter 단독 단축 — 약 33%)
#   ※ v1 완료 → ENABLE_IA=0 (기본 SKIP)
# ================================================================
if [ "${ENABLE_IA}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 I-A: n_iter=4000  (n_iter 단독 단축 ~33%)"
    echo "  n_iter=4000  batch=4096  lr=5e-3  gamma=0.33"
    echo "  비교: ff_w005_a20 (n_iter=6000) vs time_IA"
    echo "  예상 시간: ~535초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IA" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 4000 \
                --batch-size 4096 \
                --learning-rate 5e-3 \
                --gamma 0.33
        done
    done
else
    echo "  [SKIP] I-A (ENABLE_IA=0) — v1 완료, 평가에만 참여"
fi

# ================================================================
# 세트 I-B: n_iter=3000  (n_iter 단독 단축 — 약 50%)
#   ※ v1 완료 → ENABLE_IB=0 (기본 SKIP)
#   ※ Y, B-A, B-B와 동일 n_iter → 2×2 factorial 기준점
# ================================================================
if [ "${ENABLE_IB}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 I-B: n_iter=3000  (n_iter 단독 단축 ~50%)"
    echo "  n_iter=3000  batch=4096  lr=5e-3  gamma=0.33"
    echo "  비교: ff_w005_a20 (n_iter=6000) vs time_IB"
    echo "  참고: B-A, B-B, Y와 n_iter 동일 → batch/lr 효과 분리 기준점"
    echo "  예상 시간: ~400초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IB" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 3000 \
                --batch-size 4096 \
                --learning-rate 5e-3 \
                --gamma 0.33
        done
    done
else
    echo "  [SKIP] I-B (ENABLE_IB=0) — v1 완료, 평가에만 참여"
fi

# ================================================================
# 세트 I-C: n_iter=2000  (n_iter 단독 단축 — 약 67%)
#   ※ v1 완료 → ENABLE_IC=0 (기본 SKIP)
# ================================================================
if [ "${ENABLE_IC}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 I-C: n_iter=2000  (n_iter 단독 단축 ~67%)"
    echo "  n_iter=2000  batch=4096  lr=5e-3  gamma=0.33"
    echo "  비교: ff_w005_a20 (n_iter=6000) vs time_IC"
    echo "  주의: 공격적 단축 — 성능 손실 임계점 확인 목적"
    echo "  예상 시간: ~265초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_IC" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 2000 \
                --batch-size 4096 \
                --learning-rate 5e-3 \
                --gamma 0.33
        done
    done
else
    echo "  [SKIP] I-C (ENABLE_IC=0) — v1 완료, 평가에만 참여"
fi

# ================================================================
# 세트 B-A: n_iter=3000  batch=8192  lr=1e-2  (Linear Scaling)
#
#   목적: 배치 2배 + n_iter 절반 + lr 비례 증가 (Linear Scaling Rule).
#         I-B(n_iter=3000, batch=4096)와 비교하면 배치 증가의 순효과 확인.
#         lr=1e-2는 BASE 대비 2배 — 불안정할 경우 B-B와 비교.
#   변경: --batch-size 8192 / --n-iter 3000 / --learning-rate 1e-2
# ================================================================
if [ "${ENABLE_BA}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 B-A: batch=8192, n_iter=3000, lr=1e-2  (Linear Scaling)"
    echo "  비교: I-B (batch=4096, lr=5e-3) vs B-A → batch+lr 조합 효과"
    echo "  비교: B-B (lr=5e-3)             vs B-A → lr scaling 단독 효과"
    echo "  비교: Y   (batch=4096, lr=8e-3) vs B-A → batch 단독 효과 (lr↑ 고정)"
    echo "  주의: lr=1e-2는 미탐색 영역 — 불안정 시 B-B 결과와 비교"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_BA" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 3000 \
                --batch-size 8192 \
                --learning-rate 1e-2 \
                --gamma 0.33 \
                --inference-batch-size 16384
        done
    done
else
    echo "  [SKIP] B-A (ENABLE_BA=0)"
fi

# ================================================================
# 세트 B-B: n_iter=3000  batch=8192  lr=5e-3  (lr 고정 비교군)
#
#   목적: 배치 2배 + n_iter 절반, lr은 그대로 유지.
#         B-A(lr=1e-2)와 비교하면 lr linear scaling의 유효성 분리.
#         I-B(batch=4096)와 비교하면 배치 크기 증가 자체의 효과 분리.
#   변경: --batch-size 8192 / --n-iter 3000 / --learning-rate 5e-3 (유지)
# ================================================================
if [ "${ENABLE_BB}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 B-B: batch=8192, n_iter=3000, lr=5e-3  (lr 고정 비교군)"
    echo "  비교: I-B (batch=4096, lr=5e-3) vs B-B → batch 단독 효과"
    echo "  비교: B-A (lr=1e-2)             vs B-B → lr scaling 효과 분리"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_BB" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 3000 \
                --batch-size 8192 \
                --learning-rate 5e-3 \
                --gamma 0.33 \
                --inference-batch-size 16384
        done
    done
else
    echo "  [SKIP] B-B (ENABLE_BB=0)"
fi

# ================================================================
# 세트 Z: n_iter=3000  batch=4096  lr=5e-3  gamma=0.1
#
#   목적: I-B와 n_iter/batch/lr 완전 동일, gamma만 0.33 → 0.1로 강화.
#         짧은 학습(n_iter=3000)에서 lr을 더 공격적으로 낮추면
#         마지막 구간(iter 2700~3000)의 fine-tuning이 개선되는지 확인.
#         gamma=0.1: milestone마다 lr을 10배 감소
#           → iter 1500: lr=5e-3 → 5e-4
#           → iter 2250: lr=5e-4 → 5e-5
#           → iter 2700: lr=5e-5 → 5e-6
#         (vs gamma=0.33: 5e-3 → 1.65e-3 → 5.4e-4 → 1.8e-4)
#   변경: --gamma 0.1
# ================================================================
if [ "${ENABLE_Z}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 Z: n_iter=3000, batch=4096, lr=5e-3, gamma=0.1"
    echo "  비교: I-B (gamma=0.33) vs Z (gamma=0.1) → lr decay 강도 단독 효과"
    echo "  목적: 짧은 학습에서 강한 decay가 수렴에 유리한지 확인"
    echo "  예상 시간: ~400초/케이스  (I-B와 동일)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_Z" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 3000 \
                --batch-size 4096 \
                --learning-rate 5e-3 \
                --gamma 0.1
        done
    done
else
    echo "  [SKIP] Z (ENABLE_Z=0)"
fi

# ================================================================
# 세트 Y: n_iter=3000  batch=4096  lr=8e-3
#
#   목적: I-B(lr=5e-3)와 B-A(batch=8192,lr=1e-2) 사이에서
#         lr만 올렸을 때의 순효과 분리.
#         2×2 factorial 구조 완성:
#           batch=4096, lr=5e-3 → I-B
#           batch=4096, lr=8e-3 → Y     ← 이 세트
#           batch=8192, lr=5e-3 → B-B
#           batch=8192, lr=1e-2 → B-A
#         lr=8e-3은 B-A(lr=1e-2)의 80% 수준 — 안정성 우선 선택.
#   변경: --learning-rate 8e-3
# ================================================================
if [ "${ENABLE_Y}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 Y: n_iter=3000, batch=4096, lr=8e-3"
    echo "  비교: I-B (lr=5e-3) vs Y (lr=8e-3) → lr 단독 증가 효과"
    echo "  비교: B-B (batch=8192, lr=5e-3) vs Y → batch vs lr 트레이드오프"
    echo "  비교: B-A (batch=8192, lr=1e-2) vs Y → batch 단독 효과 (lr↑ 고정)"
    echo "  목적: 2×2 factorial 완성 → lr과 batch의 상호작용 분리"
    echo "  예상 시간: ~400초/케이스  (I-B와 동일)"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_Y" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 3000 \
                --batch-size 4096 \
                --learning-rate 8e-3 \
                --gamma 0.33
        done
    done
else
    echo "  [SKIP] Y (ENABLE_Y=0)"
fi

# ================================================================
# 세트 X: n_iter=5000  batch=4096  lr=5e-3
#
#   목적: I-A(4000)와 BASE(6000) 사이 보간점.
#         I-A에서 성능 손실이 유의미할 경우 "4000~6000 사이 어딘가가
#         임계점"이라는 결론을 좁히기 위한 세트.
#         I-A 손실이 미미하면 → X도 미미할 가능성 높음 (확인용).
#         I-A 손실이 크면    → X로 임계점 구간을 4000~5000으로 좁힘.
#   변경: --n-iter 5000
# ================================================================
if [ "${ENABLE_X}" -eq 1 ]; then
    echo ""
    echo "################################################################"
    echo "  세트 X: n_iter=5000, batch=4096, lr=5e-3"
    echo "  비교: I-A (n_iter=4000) vs X (n_iter=5000) vs BASE (n_iter=6000)"
    echo "  목적: n_iter 단축 임계점 구간 좁히기"
    echo "  예상 시간: ~665초/케이스"
    echo "################################################################"

    for PATIENT in "${PATIENTS[@]}"; do
        for MODALITY in "${MODALITIES[@]}"; do
            run_case "time_X" "${PATIENT}" "${MODALITY}" \
                ${BASE_ARGS} \
                --n-iter 5000 \
                --batch-size 4096 \
                --learning-rate 5e-3 \
                --gamma 0.33
        done
    done
else
    echo "  [SKIP] X (ENABLE_X=0)"
fi

# ================================================================
# 평가 실행
# ff_w005_a20, I-A, I-B, I-C : 기존 결과 파일 재사용 (재실행 없음)
# 총 9세트 비교
# ================================================================
echo ""
echo "################################################################"
echo "  평가 시작: evaluate_sweep.py"
echo "  총 9세트: ff_w005_a20 / I-A / I-B / I-C / B-A / B-B / Z / Y / X"
echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "################################################################"

python evaluate_sweep.py \
    --exp-tags ff_w005_a20 \
               time_IA \
               time_IB \
               time_IC \
               time_BA \
               time_BB \
               time_Z \
               time_Y \
               time_X \
    --patients 003 026 030 040 060 \
    --modalities flair t2 t1ce \
    --output-csv eval_time_sweep.csv

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  ❌ 평가 스크립트 실패 (exit=${EVAL_EXIT})"
else
    echo "  ✅ 평가 완료 → eval_time_sweep.csv"
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
