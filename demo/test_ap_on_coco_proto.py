import argparse, torch, torch.nn.functional as F
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.inference import load_model
from groundingdino.datasets.coco_eval import CocoEvaluator, CocoGTDataset

def build_text_only_protos(model, captions):
    tok = model.tokenizer(captions, padding="longest", return_tensors="pt").to("cuda")
    out = model.bert(**tok)
    txt = model.feat_map(out["last_hidden_state"])        # (B,T,D)
    mask = tok["attention_mask"].unsqueeze(-1).float()    # (B,T,1)
    proto = (txt * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # (B,D)
    return F.normalize(proto, dim=-1)                     # (C,D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config",required=True)
    ap.add_argument("-p","--ckpt",required=True)
    ap.add_argument("--anno_path",required=True)
    ap.add_argument("--image_dir",required=True)
    ap.add_argument("--caption",required=True,help="类别短语，建议用 . 分隔，如 'person . dog .'")
    args = ap.parse_args()

    model = load_model(args.config, args.ckpt).cuda().eval()
    # 规范 caption：切分并统一为以点号结尾的短语
    caps = [s.strip()+" ." for s in args.caption.split(".") if s.strip()]
    class_protos = build_text_only_protos(model, caps).cuda()  # (C,D)

    ds = CocoGTDataset(args.image_dir, args.anno_path, caps)
    ev = CocoEvaluator(ds.coco, iou_types=["bbox"])

    with torch.no_grad():
        for im, tgt, cap in ds:          # 与仓库自带的 test_ap_on_coco.py 一致
            samples = nested_tensor_from_tensor_list([im.cuda()])
            out = model(samples, targets=None, captions=[" ".join(caps)], class_protos=class_protos)
            logits = out.get("pred_logits_fused", out["pred_logits"])  
            boxes  = out["pred_boxes"]
            results = ds.postprocess(logits, boxes)   
            ev.update(results)

    ev.synchronize_between_processes()
    ev.accumulate()
    ev.summarize()

if __name__ == "__main__":
    main()
