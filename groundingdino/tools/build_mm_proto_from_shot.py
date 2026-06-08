# groundingdino/tools/build_mm_proto_from_shot.py
import argparse
import json
import os

import torch
from pycocotools.coco import COCO

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.models.proto_from_support import build_mm_prototypes


def load_support_items(ann_path, img_root):
    """
    兼容两种格式：
    1) 我们自己 make_support_json.py 生成的 list[{"image": "...", "boxes": [...], "labels": [...]}]
    2) 数据集自带的 COCO 风格 1_shot.json / 5_shot.json / 10_shot.json
    """
    with open(ann_path, "r") as f:
        data = json.load(f)

    # 情况 1：已经是 support_items 列表
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "image" in data[0]:
        return data

    # 情况 2：COCO 格式：{images, annotations, categories}
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        coco = COCO(ann_path)
        imgid2info = {img["id"]: img for img in coco.dataset["images"]}

        support_items = []
        # 按 image 分组 annotation
        imgid2anns = {}
        for ann in coco.dataset["annotations"]:
            imgid2anns.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in imgid2anns.items():
            info = imgid2info[img_id]
            file_name = info["file_name"]
            img_path = os.path.join(img_root, file_name)

            boxes = []
            labels = []
            for a in anns:
                x, y, w, h = a["bbox"]
                boxes.append([x, y, x + w, y + h])  # 转成 xyxy
                labels.append(a["category_id"])

            support_items.append({
                "image": img_path,
                "boxes": boxes,
                "labels": labels,
            })

        return support_items

    raise ValueError(f"Unsupported support annotation format in {ann_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="GroundingDINO config path")
    parser.add_argument("-p", "--checkpoint", required=True, help="model checkpoint (.pth)")
    parser.add_argument("--img_root", required=True, help="support images root directory")
    parser.add_argument("--support_ann", required=True, help="1_shot / 5_shot / 10_shot json")
    parser.add_argument("--out", required=True, help="output proto .pt path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1) 建模型 + 加权重
    # cfg = SLConfig.fromfile(args.config)
    # cfg.device = args.device
    # model, _, _ = build_model(cfg)

    # 1) 建模型 + 加权重
    cfg = SLConfig.fromfile(args.config)
    cfg.device = args.device

    built = build_model(cfg)
    # 有的版本返回 (model, criterion, postprocessors)
    # 有的版本只返回 model，这里做个兼容
    if isinstance(built, tuple):
        model = built[0]
    else:
        model = built

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)), strict=False)
    model.to(device)
    model.eval()



    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)), strict=False)
    model.to(device)
    model.eval()

    # 2) 读 support json -> support_items 列表
    support_items = load_support_items(args.support_ann, args.img_root)
    print(f"Loaded {len(support_items)} support items from {args.support_ann}")

    # 3) 调用 proto_from_support 里的多模态原型构建函数
    with torch.no_grad():
        proto_dict = build_mm_prototypes(model, support_items, device=device, normalize=True)

    # 4) 存成 .pt
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # torch.save({"proto": proto_dict}, args.out)
    # print(f"Saved proto to {args.out}")
        # 4) 存成 .pt（注意：直接存 dict，本身就包含 "P" 和 "proto_meta"）
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(proto_dict, args.out)
    print(f"Saved proto to {args.out}")


if __name__ == "__main__":
    main()
