# step1_inspect_v2.py
import torch
import glob
import os

RECON_ROOT = "/dshome/ddualab/dongnyeok/NeSVoR/recon_gating"

pt_files = sorted(glob.glob(f"{RECON_ROOT}/**/*_model.pt", recursive=True))
print(f"총 pt 파일 수: {len(pt_files)}\n")

# 처음 3개 키 구조 상세 출력
for pt_path in pt_files[:3]:
    name = os.path.basename(pt_path)
    ckpt = torch.load(pt_path, map_location="cpu")
    print(f"=== {name} ===")
    print(f"  최상위 키: {list(ckpt.keys())}")

    # 'model' 키 안이 실제 state_dict
    sd = ckpt["model"]
    print(f"  model 타입: {type(sd)}")

    # state_dict인지 OrderedDict인지 확인
    if hasattr(sd, "keys"):
        all_keys = list(sd.keys())
        print(f"  model 내 전체 키 수: {len(all_keys)}")
        print(f"  model 내 키 샘플 (앞 10개):")
        for k in all_keys[:10]:
            print(f"    {k}: {sd[k].shape if hasattr(sd[k], 'shape') else type(sd[k])}")
        
        # level_weights / gating 관련 키
        lw_keys = [k for k in all_keys if "level_weight" in k or "gating" in k.lower()]
        print(f"  gating 관련 키: {lw_keys}")
        for k in lw_keys:
            t = sd[k]
            import torch.nn.functional as F
            weights = F.softmax(t.float(), dim=0) * t.shape[0]
            print(f"    {k}: shape={t.shape}")
            print(f"      raw logits : {t.detach().cpu().tolist()}")
            print(f"      softmax*N  : {weights.detach().cpu().tolist()}")
    else:
        # sd가 nn.Module 객체인 경우
        print("  model이 nn.Module — state_dict() 호출")
        all_keys = list(sd.state_dict().keys())
        print(f"  키 샘플: {all_keys[:10]}")
        lw_keys = [k for k in all_keys if "level_weight" in k or "gating" in k.lower()]
        print(f"  gating 관련 키: {lw_keys}")
    print()

# 전체 파일에서 n_levels 분포 확인
print("=== 전체 파일 n_levels 분포 확인 ===")
n_levels_counter = {}
failed = []
for pt_path in pt_files:
    try:
        ckpt = torch.load(pt_path, map_location="cpu")
        sd = ckpt["model"]
        if hasattr(sd, "keys"):
            lw_keys = [k for k in sd.keys() if "level_weight" in k]
        else:
            lw_keys = [k for k in sd.state_dict().keys() if "level_weight" in k]
        
        if lw_keys:
            key = lw_keys[0]
            tensor = sd[key] if hasattr(sd, "keys") else sd.state_dict()[key]
            n = tensor.shape[0]
            n_levels_counter[n] = n_levels_counter.get(n, 0) + 1
        else:
            failed.append(os.path.basename(pt_path))
    except Exception as e:
        failed.append(f"{os.path.basename(pt_path)}: {e}")

print(f"n_levels 분포: {n_levels_counter}")
if failed:
    print(f"level_weights 키 없거나 오류 ({len(failed)}건): {failed[:5]}{'...' if len(failed)>5 else ''}")
