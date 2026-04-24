# step1_inspect.py
# 목적: pt 파일 구조 확인 (state_dict 키, level_weights shape, n_levels 분포)
import torch
import glob
import os

RECON_ROOT = "/dshome/ddualab/dongnyeok/NeSVoR/recon_gating"

pt_files = sorted(glob.glob(f"{RECON_ROOT}/**/*_model.pt", recursive=True))
print(f"총 pt 파일 수: {len(pt_files)}\n")

# 처음 3개만 키 구조 출력
for pt_path in pt_files[:3]:
    name = os.path.basename(pt_path)
    ckpt = torch.load(pt_path, map_location="cpu")
    print(f"=== {name} ===")
    print(f"  최상위 키: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    
    # state_dict 찾기
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
    else:
        sd = ckpt.state_dict()
    
    # level_weights 관련 키 찾기
    lw_keys = [k for k in sd.keys() if "level_weight" in k or "gating" in k.lower()]
    print(f"  gating 관련 키: {lw_keys}")
    for k in lw_keys:
        print(f"    {k}: shape={sd[k].shape}, values={sd[k].detach().cpu()[:5]}...")
    print()

# n_levels 분포 확인 (전체 파일)
n_levels_counter = {}
for pt_path in pt_files:
    try:
        ckpt = torch.load(pt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
        lw_keys = [k for k in sd.keys() if "level_weight" in k]
        if lw_keys:
            n = sd[lw_keys[0]].shape[0]
            n_levels_counter[n] = n_levels_counter.get(n, 0) + 1
    except Exception as e:
        print(f"  오류: {pt_path} → {e}")

print(f"\nn_levels 분포: {n_levels_counter}")
