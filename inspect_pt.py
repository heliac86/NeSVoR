# inspect_pt.py
# 목적: NeSVoR .pt 체크포인트 구조 확인 + N_LEVELS 자동 추출

import torch

# ── 경로를 여기에 채워넣으세요 ──────────────────────────────────
PT_PATH = "/dshome/ddualab/dongnyeok/NeSVoR/recon_G14s/BraTS20_Training_003/003_flair_4x5_G14s.pt"
# ────────────────────────────────────────────────────────────────

assert PT_PATH, "PT_PATH를 채워넣어 주세요."

ckpt = torch.load(PT_PATH, map_location="cpu")

# ── 1. 최상위 키 확인 ──────────────────────────────────────────
print("=" * 60)
print("[ 1. 최상위 키 (ckpt.keys()) ]")
print("=" * 60)
for k, v in ckpt.items():
    if hasattr(v, "shape"):
        print(f"  {k:<30} shape={v.shape}  dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k:<30} dict  ({len(v)} keys)")
    else:
        print(f"  {k:<30} {type(v).__name__}  = {v}")

# ── 2. model state_dict 키 중 INR 관련만 출력 ──────────────────
print("\n" + "=" * 60)
print("[ 2. model state_dict — inr.* 관련 키 ]")
print("=" * 60)
if "model" in ckpt and isinstance(ckpt["model"], dict):
    sd = ckpt["model"]
    inr_keys = [k for k in sd.keys() if k.startswith("inr.")]
    for k in inr_keys:
        v = sd[k]
        if hasattr(v, "shape"):
            print(f"  {k:<55} shape={list(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<55} {v}")

# ── 3. N_LEVELS 자동 추출 시도 ─────────────────────────────────
print("\n" + "=" * 60)
print("[ 3. N_LEVELS 자동 추출 ]")
print("=" * 60)

n_levels_found = None

# 방법 A: encoding 파라미터에서 추출 (tinycudann 방식)
if "model" in ckpt:
    sd = ckpt["model"]
    enc_keys = [k for k in sd.keys() if "encoding" in k and "params" in k]
    if enc_keys:
        print(f"  encoding params 키: {enc_keys[0]}")
        print(f"  shape: {list(sd[enc_keys[0]].shape)}")

# 방법 B: gating_net 마지막 레이어 출력 차원에서 추출
if "model" in ckpt:
    sd = ckpt["model"]
    gating_out_keys = [k for k in sd.keys()
                       if "gating_net" in k and "weight" in k]
    if gating_out_keys:
        last_key = sorted(gating_out_keys)[-1]   # 마지막 Linear weight
        shape    = list(sd[last_key].shape)
        print(f"  gating_net 마지막 레이어 키  : {last_key}")
        print(f"  shape (out_features, in_features): {shape}")
        n_levels_found = shape[0]                # out_features = n_levels
        print(f"  → 추정 N_LEVELS = {n_levels_found}")

# 방법 C: args에서 직접 읽기
if "args" in ckpt:
    args = ckpt["args"]
    if hasattr(args, "n_levels"):
        print(f"  args.n_levels = {args.n_levels}")
        n_levels_found = args.n_levels
    else:
        print("  args 객체에 n_levels 속성 없음")
        print(f"  args 속성 목록: {[a for a in dir(args) if not a.startswith('_')]}")

print("\n" + "=" * 60)
print(f"  최종 추정 N_LEVELS = {n_levels_found}")
print("=" * 60)
