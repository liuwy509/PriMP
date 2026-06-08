import argparse, os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pycocotools.coco import COCO
from groundingdino.util.inference import load_model

# ---------- robust tokenizer ----------
def get_tokenizer():
    try:
        from transformers import BertTokenizerFast
    except Exception as e:
        raise RuntimeError("transformers 未安装或不可用") from e

    # 1) 环境变量优先生效（离线本地目录）
    for k in ("GDINO_BERT_DIR", "BERT_LOCAL_DIR", "BERT_DIR"):
        v = os.getenv(k, "").strip()
        if v and os.path.isdir(v):
            try:
                return BertTokenizerFast.from_pretrained(v)
            except Exception:
                pass
    # 2) 本地缓存（离线）
    return BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)

def get_attr_by_path(root, path: str):
    obj = root
    for name in path.split("."):
        if not hasattr(obj, name): return None
        obj = getattr(obj, name)
    return obj

def try_find_text_proj(model, t_dim, v_dim):
    """
    在模型里尽量找到“文本→视觉”的线性投影层。
    优先：名字含 text/lang 且 in=t_dim, out=v_dim 的 nn.Linear。
    其次：常见路径别名。
    """
    # 常见路径优先探测
    for path in [
        "language_model.text_proj",
        "language_model.text_projection",
        "text_proj",
        "text_projection",
    ]:
        mod = get_attr_by_path(model, path)
        if isinstance(mod, nn.Linear) and mod.in_features==t_dim and mod.out_features==v_dim:
            print(f"[PROJ] use '{path}' ({t_dim}->{v_dim})")
            return mod

    # 全局扫描一遍所有子模块
    cands = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.in_features==t_dim and m.out_features==v_dim:
            score = 0
            lname = name.lower()
            if "text" in lname or "lang" in lname:
                score += 10
            if "proj" in lname or "projection" in lname:
                score += 5
            cands.append((score, name, m))
    if cands:
        cands.sort(key=lambda x: (-x[0], x[1]))
        print(f"[PROJ] auto-picked '{cands[0][1]}' ({t_dim}->{v_dim})")
        return cands[0][2]
    return None

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Build text-projected class prototypes aligned to GDINO space")
    ap.add_argument("--ann", required=True, help="COCO-format ann (取VOC类别及id顺序)")
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("-p","--ckpt",   required=True)
    ap.add_argument("--out", default="outputs/prototypes/text_prototypes.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.config, args.ckpt).to(device).eval()
    tok    = get_tokenizer()

    coco = COCO(args.ann)
    cid_list = sorted(coco.getCatIds())
    names    = [c["name"] for c in coco.loadCats(cid_list)]
    texts    = [f"a photo of a {n}" for n in names]

    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # 文本编码器输出 [CLS]
    txt_enc = getattr(model, "bert", None) or getattr(model, "text_encoder", None)
    if txt_enc is None:
        raise RuntimeError("模型缺少 bert/text_encoder，无法构建文本原型")
    out = txt_enc(**enc)
    if hasattr(out, "last_hidden_state"):
        t = out.last_hidden_state[:, 0, :]     # [C, Tdim]
    elif isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
        t = out[1]                             # [C, Tdim]
    else:
        t = out[0][:, 0, :]                    # [C, Tdim]

    Tdim = t.shape[-1]
    Vdim = getattr(model, "hidden_dim", getattr(model, "d_model", 256))

    # 文本->视觉投影
    proj = try_find_text_proj(model, Tdim, Vdim)
    if proj is None:
        print("[PROJ][WARN] 未在模型中找到匹配的文本投影层；使用临时线性映射")
        W = torch.empty(Tdim, Vdim, device=device)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        v = t @ W
    else:
        v = proj.to(device)(t)

    P = F.normalize(v, dim=-1).float().cpu()   # [C, Vdim]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"P": P, "row2cid": cid_list}, args.out)
    print(f"[OK] saved text-projected P -> {args.out} shape={tuple(P.shape)}; C={len(cid_list)}")

if __name__ == "__main__":
    main()
