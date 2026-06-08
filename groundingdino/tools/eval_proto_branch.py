import os, sys, json, re
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

def norm_img_id(x):
    return str(x)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import math
import time

# torchvision NMS（后融合阶段用）
try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None
    print("[WARN] torchvision.ops.nms not available; post-fusion NMS will be skipped.")

from groundingdino.util.inference import load_model, load_image
from groundingdino.util.inference import predict as gdino_predict

# ----------------------------------------------------------------------
# HOTFIX：在 argparse 之前手动吃掉 --coord_mode，避免“未注册参数”报错
# ----------------------------------------------------------------------
COORD_MODE = "file2ann"
for key in ("--coord_mode", "--coord-mode"):
    if key in sys.argv:
        try:
            i = sys.argv.index(key)
            COORD_MODE = sys.argv[i + 1].lower()
            del sys.argv[i:i + 2]
        except Exception:
            pass

# ---------- 文本规范化 & 常见别名 ----------
_ART_RE  = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNC_RE = re.compile(r"[.,;:!?\(\)\[\]\{\}]+")
_WS_RE   = re.compile(r"\s+")

_ALIASES = {
    "sofa": "couch",
    "tv monitor": "tv", "tvmonitor": "tv", "television": "tv",
    "cellphone": "cell phone", "mobile phone": "cell phone",
    "motorbike": "motorcycle",
    "air plane": "airplane",
    "trafficlight": "traffic light",
    "parkingmeter": "parking meter",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
    "wineglass": "wine glass",
    "hairdryer": "hair drier",
    "fridge": "refrigerator",
    "microwave oven": "microwave",
    "hand bag": "handbag",
    "remote control": "remote", "remote-control": "remote", "remotecontrol": "remote",
    "tennis racquet": "tennis racket",
    "people":"person","man":"person","woman":"person","boy":"person","girl":"person",
    "ball":"sports ball","table":"dining table","plant":"potted plant",
    "hydrant":"fire hydrant","stopsign":"stop sign",
    "ski":"skis","racket":"tennis racket","racquet":"tennis racket",
    "mobile phone":"cell phone","phone":"cell phone",
}

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = _ART_RE.sub(" ", s)
    s = _PUNC_RE.sub(" ", s)
    s = s.replace("photo of", " ")        # 去掉 photo of 前缀
    s = s.replace("-", " ").replace("_", " ")
    s = _WS_RE.sub(" ", s)
    s = _ALIASES.get(s, s)
    return s

def _norm_ns(s: str) -> str:
    return _norm(s).replace(" ", "")

def prob_to_logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))

def iou_single_vs_many(b1: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(b1[0], B[:,0])
    y1 = torch.maximum(b1[1], B[:,1])
    x2 = torch.minimum(b1[2], B[:,2])
    y2 = torch.minimum(b1[3], B[:,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (b1[2]-b1[0]).clamp(min=0) * (b1[3]-b1[1]).clamp(min=0)
    area2 = (B[:,2]-B[:,0]).clamp(min=0) * (B[:,3]-B[:,1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union

def soft_nms_gaussian(boxes: torch.Tensor,
                      scores: torch.Tensor,
                      iou_thr: float = 0.7,
                      sigma: float = 0.5,
                      score_thresh: float = 1e-5) -> torch.Tensor:
    N = boxes.size(0)
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(N, device=boxes.device)
    keep = []
    while idxs.numel() > 0:
        max_local = torch.argmax(scores[idxs])
        cur = idxs[max_local]
        keep.append(cur)
        if idxs.numel() == 1:
            break
        rest = torch.cat([idxs[:max_local], idxs[max_local+1:]], dim=0)
        ious = iou_single_vs_many(boxes[cur], boxes[rest])
        decay = torch.exp(- (ious * ious) / max(1e-6, sigma))
        mask_hiou = ious >= float(iou_thr)
        scores[rest[mask_hiou]] *= decay[mask_hiou]
        idxs = idxs[scores[idxs] > score_thresh]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long, device=boxes.device)

# -----------------------  词库加载 & prompt 组装  -----------------------
def _safe_load_lexicon(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[LEX][WARN] file not found: {path}")
        return None
    txt = p.read_text(encoding="utf-8")
    data = None
    try:
        import yaml  # noqa
        data = yaml.safe_load(txt)
    except Exception:
        try:
            data = json.loads(txt)
        except Exception as e:
            print(f"[LEX][WARN] failed to parse YAML/JSON: {e}")
            return None
    if not isinstance(data, dict) or "classes" not in data:
        print("[LEX][WARN] malformed lexicon (expect dict with key 'classes').")
        return None
    out = {}
    for item in data.get("classes", []):
        try:
            cid  = int(item.get("id"))
            name = str(item.get("name", "")).strip()
        except Exception:
            continue
        syn = item.get("syn", item.get("synonyms", [])) or []
        if isinstance(syn, str):
            syn = [syn]
        syn = [s for s in [str(x).strip() for x in syn] if s]
        definition = item.get("def", item.get("definition", "")) or ""
        definition = str(definition).strip()
        out[cid] = {"name": name, "syn": syn, "def": definition}
    print(f"[LEX] loaded classes: {len(out)} from {path}")
    return out

def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = _WS_RE.split(text.strip())
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

def _build_tagged_items(cid: int, name: str, syns, definition: str,
                        use_syn_k: int, use_def: bool, def_max_words: int,
                        tpl: str = "plain"):
    items = []
    def _wrap(nm):
        if tpl == "photo":
            return f"[{cid}] a photo of a {nm}"
        return f"[{cid}] {nm}"

    if name:
        items.append(_wrap(name))
    if use_syn_k and syns:
        for s in syns[:use_syn_k]:
            s = s.strip()
            if s:
                items.append(_wrap(s))
    if use_def and definition:
        d = _truncate_words(definition, def_max_words).strip()
        if d:
            items.append(f"[{cid}] {d}")
    return items

# ----------------------- 文本校准加载 -----------------------
def _load_text_calib(path: str, ids_sorted):
    if not path:
        return None, None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[CALIB][WARN] failed to load: {e}")
        return None, None

    alpha = ckpt.get("alpha", None)
    beta  = ckpt.get("beta", None)
    if alpha is None or beta is None:
        print("[CALIB][WARN] missing keys 'alpha'/'beta' in calib file.")
        return None, None
    alpha = torch.as_tensor(alpha).flatten().float()
    beta  = torch.as_tensor(beta).flatten().float()

    order_list = ckpt.get("class_order", ckpt.get("cids", None))
    cid2idx = None
    if order_list is not None:
        try:
            cid2idx = {int(c): i for i, c in enumerate(order_list)}
        except Exception:
            cid2idx = None

    A = torch.ones(len(ids_sorted), dtype=torch.float32)
    B = torch.zeros(len(ids_sorted), dtype=torch.float32)
    for j, cid in enumerate(ids_sorted):
        idx = cid2idx.get(int(cid), None) if cid2idx is not None else int(cid) - 1
        if idx is not None and 0 <= idx < alpha.numel():
            A[j] = alpha[idx]
        if idx is not None and 0 <= idx < beta.numel():
            B[j] = beta[idx]
    print("[CALIB] loaded per-class α/β and aligned to current category order.")
    return A, B

@torch.no_grad()
def run(cfg, ckpt, img_dir, ann, proto_path,
        lam=0.30, tau=0.07, box_th=0.30, text_th=0.25, nms_th=0.50,
        coord_mode: str = COORD_MODE,
        out_json="outputs/dets_proto_branch.json", device=None, viz_dir=None,
        gate_top1=False, gate_thr=0.80, gate_margin=0.10,
        tag_prompt=False, prompt_tpl="plain",
        soft_nms=False, soft_sigma=0.5,
        adapt_lam=False, gap_thr=0.10, gap_beta=10.0, dump_gate_stats=None,
        lexicon_path=None, use_syn=0, use_def=False, def_max_words=14,
        text_calib_path=None,
        cand_map_path=None, restrict_cand=False,
        novel_only=False, novel_ids="16,17,18,19,20",
        plain_def_k=0,
        box_fmt="cxcywh"):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_model(cfg, ckpt).to(device).eval()

    # 1) 原型矩阵
    proto_raw = torch.load(proto_path, map_location="cpu")
    proto_meta = {}
    if isinstance(proto_raw, dict):
        proto_meta = proto_raw
        for k in ("vis_embed", "proto", "P", "emb", "tensor", "prototypes", "feat", "protos"):
            if k in proto_raw:
                P = proto_raw[k]
                break
        else:
            raise ValueError("Proto file is a dict but no tensor under known keys.")
    else:
        P = proto_raw

    P = P if torch.is_tensor(P) else torch.tensor(P)
    P = F.normalize(P.float(), dim=-1)

    # 2) 类别
    coco = COCO(ann)
    catid2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ids_sorted   = sorted(catid2name)
    names_sorted = [catid2name[i] for i in ids_sorted]
    img_ids = list(coco.getImgIds())

    # 3) 词库
    lex = _safe_load_lexicon(lexicon_path) if lexicon_path else None
    id2syn = {cid: [] for cid in ids_sorted}
    id2def = {cid: "" for cid in ids_sorted}
    if lex:
        for cid in ids_sorted:
            if cid in lex:
                id2syn[cid] = lex[cid].get("syn", []) or []
                id2def[cid] = lex[cid].get("def", "") or ""

    # 4) 映射
    name2cat    = {_norm(n): i for n, i in zip(names_sorted, ids_sorted)}
    name2cat_ns = {_norm_ns(n): i for n, i in zip(names_sorted, ids_sorted)}
    if lex:
        for cid in ids_sorted:
            for s in id2syn[cid]:
                name2cat[_norm(s)] = cid
                name2cat_ns[_norm_ns(s)] = cid

    catid2row   = {cid: r for r, cid in enumerate(ids_sorted)}
    row2cat     = {r: cid for r, cid in enumerate(ids_sorted)}

    # 5) 文本校准
    A_alpha, B_beta = _load_text_calib(text_calib_path, ids_sorted)

    # 6) Two-Phase
    cand = None
    if cand_map_path:
        try:
            _m = json.load(open(cand_map_path))
            cand = {}
            for k, v in _m.items():
                try:
                    cand[int(k)] = [int(x) for x in v]
                except Exception:
                    pass
            print(f"[CAND] loaded per-image whitelist for {len(cand)} images.")
        except Exception as e:
            print("[CAND][WARN]", e)

    # 7) hook decoder 最后一层 query
    last_hs = {"feat": None}
    def _hook(module, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.transpose(0, 1)
        last_hs["feat"] = x.contiguous()

    handle = model.transformer.decoder.layers[-1].register_forward_hook(lambda m, i, o: _hook(m, i, o))

    dets = []
    phrase_hit_total = 0
    fallback_hit_total = 0
    box_total = 0
    phrase_map_ok, phrase_map_tot = 0, 0
    gate_attempt_total = 0
    gate_kept_total = 0
    gate_stats = []

    coord_mode = (coord_mode or "file2ann").lower()
    fmt_print_left = 3
    sz_print_left = 3
    postnms_print_left = 3
    viz_done = 0

    for img_id in tqdm(img_ids, desc="proto-branch"):
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, info["file_name"])
        image_source, image = load_image(img_path)

        # 候选类集合
        iter_ids   = ids_sorted
        iter_names = names_sorted
        if restrict_cand and cand:
            keep = set(cand.get(norm_img_id(img_id), []))
            if keep:
                pairs = [(cid, name) for cid, name in zip(ids_sorted, names_sorted) if cid in keep]
                if pairs:
                    iter_ids, iter_names = zip(*pairs)
                else:
                    iter_ids, iter_names = (), ()
        if len(iter_names) == 0:
            continue

        # prompt
        if tag_prompt:
            pieces = []
            for cid, name in zip(iter_ids, iter_names):
                items = _build_tagged_items(
                    cid=cid, name=name, syns=id2syn.get(cid, []), definition=id2def.get(cid, ""),
                    use_syn_k=use_syn, use_def=use_def, def_max_words=def_max_words,
                    tpl=prompt_tpl
                )
                pieces.extend(items)
            prompt = " . ".join(pieces) + " ."
        else:
            cats_for_prompt = []
            for cid, name in zip(iter_ids, iter_names):
                base = f"a photo of a {name}" if prompt_tpl == "photo" else name
                if (plain_def_k and plain_def_k > 0) and lex and use_def and (def_max_words > 0):
                    d = id2def.get(cid, "") or ""
                    if d:
                        K = min(int(plain_def_k), int(def_max_words))
                        words = _WS_RE.split(d.strip())
                        dshort = " ".join(words[:K])
                        dshort = _ART_RE.sub(" ", dshort).strip()
                        if dshort:
                            base = f"{base}; {dshort}"
                cats_for_prompt.append(base)
            prompt = " . ".join(cats_for_prompt) + " ."

        boxes, logits, phrases = gdino_predict(
            model, image, prompt, box_threshold=box_th, text_threshold=text_th
        )

        # ---- 坐标统一：关键改动：box_fmt 默认 cxcywh ----
        W_gt, H_gt = info["width"], info["height"]
        W_im, H_im = W_gt, H_gt
        try:
            if hasattr(image_source, "size") and hasattr(image_source, "getbands"):
                W_im, H_im = image_source.size
            elif hasattr(image_source, "shape"):
                H_im, W_im = int(image_source.shape[0]), int(image_source.shape[1])
        except Exception:
            pass

        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

        if boxes.numel() > 0:
            W_ref, H_ref = (W_gt, H_gt) if coord_mode == "gt_only" else (W_im, H_im)
            is_norm = boxes.max().item() <= 1.5
            fmt = str(box_fmt).lower()

            if fmt_print_left > 0:
                print(f"[FMT] mode={coord_mode} is_norm={is_norm} fmt={fmt} ref=({W_ref},{H_ref}) n={len(boxes)}")
                fmt_print_left -= 1

            if is_norm:
                if fmt == "cxcywh":
                    cx = boxes[:, 0] * float(W_ref)
                    cy = boxes[:, 1] * float(H_ref)
                    ww = boxes[:, 2] * float(W_ref)
                    hh = boxes[:, 3] * float(H_ref)
                    x1 = cx - ww / 2.0
                    y1 = cy - hh / 2.0
                    x2 = cx + ww / 2.0
                    y2 = cy + hh / 2.0
                else:  # xyxy
                    x1 = boxes[:, 0] * float(W_ref)
                    y1 = boxes[:, 1] * float(H_ref)
                    x2 = boxes[:, 2] * float(W_ref)
                    y2 = boxes[:, 3] * float(H_ref)
            else:
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            if coord_mode == "file2ann" and ((W_im != W_gt) or (H_im != H_gt)):
                sx = float(W_gt) / max(1.0, float(W_im))
                sy = float(H_gt) / max(1.0, float(H_im))
                x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy

            x1 = torch.min(x1, x2).clamp(0.0, float(W_gt - 1))
            y1 = torch.min(y1, y2).clamp(0.0, float(H_gt - 1))
            x2 = torch.max(x1 + 1e-6, x2).clamp(0.0, float(W_gt))
            y2 = torch.max(y1 + 1e-6, y2).clamp(0.0, float(H_gt))
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

        if boxes.numel() > 0 and sz_print_left > 0:
            print(f"[SZ] box_min={boxes.min().item():.1f} box_max={boxes.max().item():.1f}")
            sz_print_left -= 1

        if (last_hs["feat"] is None) or (boxes.numel() == 0):
            continue

        # decoder query 特征
        q = last_hs["feat"].to(device)
        if q.dim() == 2:
            q = q[None, ...]
        q = F.normalize(q, dim=-1)

        N = len(boxes)
        q_keep = q[:, :N, :]
        P_dev = P.to(q_keep.device, dtype=q_keep.dtype, non_blocking=True)
        proto_logits = (q_keep @ P_dev.t()[None, :, :]).squeeze(0)
        proto_logits = proto_logits / max(1e-6, float(tau))

        # phrase -> cid
        cls_ids = []
        keys_norm   = list(name2cat.keys())
        keys_ns     = list(name2cat_ns.keys())
        phrase_ns_cache = {}

        phrases_use = phrases[:N] if isinstance(phrases, (list, tuple)) else [phrases]*N
        phrase_hits = 0
        fallback_hits = 0

        for i, ph in enumerate(phrases_use):
            cid = None
            m = re.search(r"\[(\d+)\]", ph) if tag_prompt else None
            if m:
                maybe = int(m.group(1))
                if maybe in catid2row:
                    cid = maybe

            if cid is None:
                t  = _norm(ph)
                t2 = phrase_ns_cache.get(ph)
                if t2 is None:
                    t2 = _norm_ns(ph)
                    phrase_ns_cache[ph] = t2
                cid = name2cat.get(t) or name2cat_ns.get(t2)
                if cid is None:
                    best_k, best_len = None, 0
                    for k in keys_norm:
                        if k in t and len(k) > best_len:
                            best_k, best_len = k, len(k)
                    if best_k is None:
                        for k in keys_ns:
                            if k in t2 and len(k) > best_len:
                                best_k, best_len = k, len(k)
                    if best_k is not None:
                        cid = name2cat.get(best_k) or name2cat_ns.get(best_k)

            if cid is not None:
                phrase_hits += 1
            else:
                logits_i = proto_logits[i]
                probs_i  = torch.softmax(logits_i, dim=-1)
                top2     = torch.topk(probs_i, k=2)
                top1_row = int(top2.indices[0].item())
                conf1    = float(top2.values[0].item())
                conf2    = float(top2.values[1].item())
                pass_gate = (conf1 >= float(gate_thr)) and ((conf1 - conf2) >= float(gate_margin))

                if gate_top1:
                    gate_attempt_total += 1
                    if pass_gate:
                        cid = row2cat.get(top1_row, None)
                        if cid is not None:
                            gate_kept_total += 1
                            fallback_hits += 1
                else:
                    cid = row2cat.get(int(torch.argmax(probs_i).item()), None)
                    if cid is not None:
                        fallback_hits += 1

            cls_ids.append(cid)

        phrase_hit_total += phrase_hits
        fallback_hit_total += fallback_hits
        box_total += len(phrases_use)
        phrase_map_ok += phrase_hits
        phrase_map_tot += len(phrases_use)

        # 融合
        scores = torch.as_tensor(logits, device=device).float()[:N]
        scores_new = scores.clone()

        for i, cid in enumerate(cls_ids):
            if cid is None:
                continue
            row = catid2row.get(cid, None)
            if row is None or row >= P.shape[0]:
                continue

            text_logit = prob_to_logit(scores[i])

            # Gate statistics for failure analysis.
            # Here alpha_eff corresponds to the effective prototype-branch coefficient.
            proto_gap_stat = None
            text_unc_stat = None
            alpha_eff_stat = 0.0
            p_txt_stat = None
            p_proto_stat = None

            if (A_alpha is not None) and (B_beta is not None):
                a = A_alpha[row].to(text_logit.device)
                b = B_beta[row].to(text_logit.device)
                text_logit = a * text_logit + b

            proto_logit = proto_logits[i][row]

            if lam > 0:
                p_txt = torch.sigmoid(text_logit).clamp(1e-5, 1 - 1e-5)

                if adapt_lam:
                    probs_i = torch.softmax(proto_logits[i], dim=-1)
                    vals, _ = torch.topk(probs_i, k=2)
                    gap = (vals[0] - vals[1]).clamp(min=0)
                    ent = -(p_txt * torch.log(p_txt) + (1 - p_txt) * torch.log(1 - p_txt))
                    unc = ent / math.log(2.0)
                    lam_eff = lam * torch.sigmoid(gap_beta * (gap - gap_thr)) * unc

                    proto_gap_stat = float(gap.detach().cpu().item())
                    text_unc_stat = float(unc.detach().cpu().item())
                    alpha_eff_stat = float(lam_eff.detach().cpu().item())
                else:
                    lam_eff = lam
                    alpha_eff_stat = float(lam)

                p_proto = torch.sigmoid(proto_logit).clamp(1e-5, 1 - 1e-5)
                p_txt_stat = float(p_txt.detach().cpu().item())
                p_proto_stat = float(p_proto.detach().cpu().item())
                mix_prob = (1.0 - lam_eff) * p_txt + lam_eff * p_proto
                mix_prob = mix_prob.clamp(1e-5, 1 - 1e-5)
                mix_logit = torch.log(mix_prob / (1.0 - mix_prob))
            else:
                mix_logit = text_logit

            scores_new[i] = torch.sigmoid(mix_logit)

            if dump_gate_stats:
                gate_stats.append({
                    "image_id": int(img_id) if str(img_id).isdigit() else str(img_id),
                    "category_id": int(cid),
                    "row": int(row),
                    "score_before": float(scores[i].detach().cpu().item()),
                    "score_after": float(scores_new[i].detach().cpu().item()),
                    "alpha_eff": float(alpha_eff_stat),
                    "proto_gap": proto_gap_stat,
                    "text_uncertainty": text_unc_stat,
                    "p_txt": p_txt_stat,
                    "p_proto": p_proto_stat,
                    "adapt_lam": bool(adapt_lam),
                    "global_lam": float(lam),
                    "gap_thr": float(gap_thr),
                    "gap_beta": float(gap_beta),
                })

        # per-class NMS
        per_cat = {}
        for b, sn, cid in zip(boxes, scores_new.tolist(), cls_ids):
            if cid is None:
                continue
            pack = per_cat.setdefault(int(cid), {"boxes": [], "scores": []})
            pack["boxes"].append(b)
            pack["scores"].append(sn)

        final_dets = []
        kept_all = 0
        total_all = 0
        for cid, pack in per_cat.items():
            if len(pack["scores"]) == 0:
                continue
            B = torch.stack(pack["boxes"], dim=0)
            S = torch.tensor(pack["scores"], device=device)
            total_all += len(S)

            if soft_nms:
                keep = soft_nms_gaussian(B, S, iou_thr=float(nms_th), sigma=float(soft_sigma))
            else:
                keep = tv_nms(B, S, float(nms_th)) if tv_nms is not None else torch.arange(len(S), device=device)

            kept_all += int(keep.numel())
            B = B[keep]; S = S[keep]
            for b, sn in zip(B.tolist(), S.tolist()):
                x1, y1, x2, y2 = b
                w = max(1e-6, x2 - x1)
                h = max(1e-6, y2 - y1)
                final_dets.append({
                    "image_id": img_id,
                    "category_id": int(cid),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(sn)
                })

        dets.extend(final_dets)
        if postnms_print_left > 0:
            print(f"[POST-NMS] kept {kept_all}/{total_all} boxes after per-class {'Soft-' if soft_nms else ''}NMS at IoU={nms_th}")
            postnms_print_left -= 1

        # 可视化（前3张）
        if viz_dir and viz_done < 3:
            try:
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                pil_img = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(pil_img, "RGBA")
                for d in final_dets:
                    x1, y1, w, h = d["bbox"]
                    draw.rectangle([x1, y1, x1+w, y1+h], outline=(0, 255, 255, 255), width=2)
                pil_img.save(os.path.join(viz_dir, f"{norm_img_id(img_id)}.jpg"))
                viz_done += 1
            except Exception as e:
                print("[VIZ][WARN]", e)

    if phrase_map_tot > 0:
        print(f"[PHRASE MAP] matched {phrase_map_ok}/{phrase_map_tot} = {phrase_map_ok/max(1,phrase_map_tot):.2%}")
    if box_total > 0:
        cov_phrase   = 100.0 * phrase_hit_total   / box_total
        cov_fallback = 100.0 * fallback_hit_total / box_total
        cov_total    = 100.0 * (phrase_hit_total + fallback_hit_total) / box_total
        print(f"[COVER] phrase={cov_phrase:.2f}%  fallback={cov_fallback:.2f}%  total={cov_total:.2f}%")

    if gate_top1 and gate_attempt_total > 0:
        pct = 100.0 * gate_kept_total / max(1, gate_attempt_total)
        print(f"[GATE] fallback_kept={gate_kept_total}/{gate_attempt_total} = {pct:.2f}% (thr={gate_thr}, margin={gate_margin})")

    if dump_gate_stats:
        import json as _json
        from pathlib import Path as _Path
        _Path(dump_gate_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_gate_stats, "w") as f:
            _json.dump(gate_stats, f, indent=2)
        print(f"[GATE_STATS] saved {len(gate_stats)} records -> {dump_gate_stats}")

    handle.remove()

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dets, open(out_json, "w"))
    print(f"[OK] saved dets -> {out_json} (#={len(dets)})")
    print(f"[INFO] processed images: {len(img_ids)}")

    cocoDt = coco.loadRes(out_json)
    e = COCOeval(coco, cocoDt, "bbox")
    e.params.imgIds = img_ids

    if novel_only:
        try:
            novel_list = [int(x) for x in str(novel_ids).split(",") if str(x).strip() != ""]
            if len(novel_list) > 0:
                e.params.useCats = 1
                e.params.catIds = novel_list
                print(f"[EVAL] novel-only mode ON, catIds = {novel_list}")
        except Exception as ex:
            print(f"[EVAL][WARN] failed to parse novel_ids='{novel_ids}': {ex}")

    e.evaluate()
    e.accumulate()
    e.summarize()
    return e.stats.tolist()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Proto-guided branch (logit fusion + tagged text with lexicon + Soft-NMS + Two-Phase whitelist)")

    ap.add_argument("--text_calib", type=str, default=None)

    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-p", "--ckpt", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--proto", required=True)

    ap.add_argument("--lam", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--box_th", type=float, default=0.30)
    ap.add_argument("--text_th", type=float, default=0.25)
    ap.add_argument("--nms_th", type=float, default=0.50)
    ap.add_argument("--coord_mode", choices=["file2ann","gt_only"], default=COORD_MODE)

    # 关键：强制指定 gdino_predict 返回的归一化 box 格式
    ap.add_argument("--box_fmt", choices=["cxcywh","xyxy"], default="cxcywh",
                    help="format of normalized boxes returned by gdino_predict (default=cxcywh)")

    ap.add_argument("--novel_only", action="store_true")
    ap.add_argument("--novel_ids", type=str, default="16,17,18,19,20")

    ap.add_argument("--gate_top1", action="store_true")
    ap.add_argument("--gate_thr", type=float, default=0.80)
    ap.add_argument("--gate_margin", type=float, default=0.10)

    ap.add_argument("--tag_prompt", action="store_true")
    ap.add_argument("--prompt_tpl", choices=["plain","photo"], default="plain")

    ap.add_argument("--soft_nms", action="store_true")
    ap.add_argument("--soft_sigma", type=float, default=0.5)

    ap.add_argument("--adapt_lam", action="store_true")
    ap.add_argument("--gap_thr", type=float, default=0.10)
    ap.add_argument("--gap_beta", type=float, default=10.0)
    ap.add_argument("--dump_gate_stats", default=None)

    ap.add_argument("--lexicon", dest="lexicon_path", default=None)
    ap.add_argument("--use_syn", type=int, default=0)
    ap.add_argument("--use_def", action="store_true")
    ap.add_argument("--def_max_words", type=int, default=14)

    ap.add_argument("--cand_map", default=None)
    ap.add_argument("--restrict_cand", action="store_true")

    ap.add_argument("--plain_def_k", type=int, default=0)

    ap.add_argument("--viz_dir", default=None)
    ap.add_argument("--out", default="outputs/dets_proto_branch.json")
    args = ap.parse_args()

    stats = run(args.config, args.ckpt, args.img_dir, args.ann, args.proto,
                lam=args.lam, tau=args.tau, box_th=args.box_th, text_th=args.text_th,
                nms_th=args.nms_th, coord_mode=args.coord_mode,
                out_json=args.out, viz_dir=args.viz_dir,
                gate_top1=args.gate_top1, gate_thr=args.gate_thr, gate_margin=args.gate_margin,
                tag_prompt=args.tag_prompt, prompt_tpl=args.prompt_tpl,
                soft_nms=args.soft_nms, soft_sigma=args.soft_sigma,
                adapt_lam=args.adapt_lam, gap_thr=args.gap_thr, gap_beta=args.gap_beta, dump_gate_stats=args.dump_gate_stats,
                lexicon_path=args.lexicon_path, use_syn=args.use_syn, use_def=args.use_def,
                def_max_words=args.def_max_words,
                text_calib_path=args.text_calib,
                cand_map_path=args.cand_map, restrict_cand=args.restrict_cand,
                plain_def_k=args.plain_def_k,
                novel_only=args.novel_only, novel_ids=args.novel_ids,
                box_fmt=args.box_fmt)

    print("Final results:", stats)
