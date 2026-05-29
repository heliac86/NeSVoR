#!/usr/bin/env bash
# ==============================================================================
# G14 전체 실험 런처 — GPU 0, 1 병렬 실행
# 사용법:
#   DATA_ROOT=/path/to/data OUT_ROOT=/path/to/results bash experiments/run_G14_all.sh
#
# 환경변수:
#   DATA_ROOT  : 케이스 루트 디렉터리 (기본: /data/nesvor)
#   OUT_ROOT   : 결과 저장 루트 (기본: /results/G14)
#   CASE_LIST  : 케이스 ID 목록 파일 (기본: DATA_ROOT/cases.txt)
#                한 줄에 케이스 ID 하나, #으로 시작하는 줄은 주석 처리
# ==============================================================================
set -euo pipefail

export DATA_ROOT="${DATA_ROOT:-/data/nesvor}"
export OUT_ROOT="${OUT_ROOT:-/results/G14}"
export CASE_LIST="${CASE_LIST:-${DATA_ROOT}/cases.txt}"

echo "================================================================"
echo " G14 실험 시작"
echo "  DATA_ROOT  : ${DATA_ROOT}"
echo "  OUT_ROOT   : ${OUT_ROOT}"
echo "  CASE_LIST  : ${CASE_LIST}"
echo "  GPU 0      : G14_clamp02, G14_clamp05"
echo "  GPU 1      : G14_temp15,  G14_silu"
echo "================================================================"

mkdir -p "${OUT_ROOT}"

# GPU 0, 1 병렬 실행 — 각 스크립트는 내부에서 케이스를 순차 처리
bash "run_G14_gpu0.sh" &
PID0=$!
echo "[launcher] GPU 0 PID=${PID0}  (clamp02, clamp05)"

bash "run_G14_gpu1.sh" &
PID1=$!
echo "[launcher] GPU 1 PID=${PID1}  (temp15, silu)"

# 두 프로세스 모두 종료될 때까지 대기
wait ${PID0} && echo "[launcher] GPU 0 완료"
wait ${PID1} && echo "[launcher] GPU 1 완료"

echo "================================================================"
echo " G14 전체 실험 완료"
echo "================================================================"
