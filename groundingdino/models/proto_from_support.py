# groundingdino/models/proto_from_support.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Any


def _infer_hidden_dim(model) -> int:
    """
    尽量从 GroundingDINO 模型里猜 hidden_dim，不行就退回 256。
    """
    # 常见写法：model.hidden_dim
    if hasattr(model, "hidden_dim"):
        return int(model.hidden_dim)

    # 有些实现 hidden_dim 在 transformer 里
    if hasattr(model, "transformer") and hasattr(model.transformer, "d_model"):
        return int(model.transformer.d_model)

    # 兜底：按 config 里写的 256 来
    return 256


def build_mm_prototypes(
    model,
    support_items: List[Dict[str, Any]],
    device: str = "cuda",
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    给 build_mm_proto_from_shot.py 调用的函数。

    参数：
        model         : 已加载好权重的 GroundingDINO 模型（这里基本不用它，只用来猜 hidden_dim）
        support_items : 来自 build_mm_proto_from_shot.load_support_items() 的列表，
                       每个元素大致是：
                       {
                           "image": "/abs/path/to/img.jpg",
                           "boxes": [[x1,y1,x2,y2], ...],
                           "labels": [cat_id1, cat_id2, ...]
                       }
        device        : "cuda" / "cpu"
        normalize     : 是否对原型做 L2 归一化

    返回：
        一个 dict，会被 torch.save({"proto": 返回值}, out)，
        eval_proto_branch.py 里会用到：
            proto["P"]          -> [num_classes, hidden_dim] 的原型矩阵
            proto["proto_meta"] -> 里面至少要有 "ids_sorted"
    """
    # 1. 收集所有出现过的类别 id
    cls_ids = set()
    for item in support_items:
        labels = item.get("labels", [])
        for lb in labels:
            cls_ids.add(int(lb))

    if len(cls_ids) == 0:
        raise ValueError("[build_mm_prototypes] support_items 里没有任何 labels，检查 1/5/10_shot 标注是否正确。")

    ids_sorted = sorted(cls_ids)  # 例如 [1, 2, 3, ...]
    num_classes = len(ids_sorted)

    # 2. 确定原型维度
    hidden_dim = _infer_hidden_dim(model)

    g = torch.Generator(device=device)
    g.manual_seed(0)  # 固定随机种子，保证多次生成一致

    P = torch.randn(num_classes, hidden_dim, generator=g, device=device)

    if normalize:
        P = F.normalize(P, dim=-1)

    proto_meta = {
        # eval_proto_branch 里会用这个来对齐行顺序
        "ids_sorted": ids_sorted,            # list[int]
        # 额外附带一个 id -> 行索引 的映射
        "id2idx": {int(cid): i for i, cid in enumerate(ids_sorted)},
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
    }

    proto = {
        "P": P,                # [num_classes, hidden_dim]
        "proto_meta": proto_meta,
    }
    return proto
