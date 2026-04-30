"""
eval_hsm_dataset.py
===================
evaluate_batch_v2.py의 평가 로직을 재사용하는 래퍼.
evaluate_batch_v2.py의 설정부(METHODS, RESULT_CSV)를 이 파일에서 오버라이드.

대상 세트:
  IC      : alpha=0.99, start=0  (기존 baseline)
  hsm_I   : alpha=0.80, start=0
  hsm_M   : alpha=0.75, start=0
  hsm_L   : alpha=0.70, start=0
  hsm_N   : alpha=0.60, start=0
  hsm_O   : alpha=0.50, start=0
  hsm_P   : alpha=0.40, start=0
"""

import evaluate_batch_v2 as ev

# ── 평가 대상 오버라이드 ──────────────────────────────────────
RECON_BASE = "/dshome/ddualab/dongnyeok/NeSVoR"

ev.METHODS = {
    "IC"    : f"{RECON_BASE}/recon_IC",
    "hsm_I" : f"{RECON_BASE}/recon_HSM_hsm_I",
    "hsm_M" : f"{RECON_BASE}/recon_HSM_hsm_M",
    "hsm_L" : f"{RECON_BASE}/recon_HSM_hsm_L",
    "hsm_N" : f"{RECON_BASE}/recon_HSM_hsm_N",
    "hsm_O" : f"{RECON_BASE}/recon_HSM_hsm_O",
    "hsm_P" : f"{RECON_BASE}/recon_HSM_hsm_P",
}

ev.RESULT_CSV = "/dshome/ddualab/dongnyeok/eval_results/eval_hsm_dataset.csv"

# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ev.main()
