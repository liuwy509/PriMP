# groundingdino/util/syn_def_prompt.py
from typing import Dict, List
import yaml, re

def _clean(s: str) -> str:
    s = s.strip()
    # 控长度 & 去奇怪符号，避免超长 token 化
    s = re.sub(r"\s+", " ", s)
    return s

def load_syn_def(yaml_path: str) -> Dict[str, Dict[str, List[str] or str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    norm = {}
    for k, v in data.items():
        syn = v.get("syn", []) or []
        syn = [_clean(x) for x in syn][:2]      # 每类最多保留2个同义词
        d = _clean(v.get("def", ""))            # 定义一小句（≤15词）
        norm[k.lower()] = {"syn": syn, "def": d}
    return norm

def build_prompt(class_names: List[str],
                 syn_def: Dict[str, Dict[str, List[str] or str]],
                 tpl: str = "[CLS] ([SYN]) - [DEF]",
                 sep: str = " . ") -> str:
    parts = []
    for cls in class_names:
        key = cls.lower().strip()
        info = syn_def.get(key, {"syn": [], "def": ""})
        syn = ", ".join(info.get("syn", []) or [])
        defn = info.get("def", "") or ""
        text = tpl
        text = text.replace("[CLS]", cls)
        text = text.replace("[SYN]", syn).replace("()", "")  # 避免空括号
        if "[DEF]" in text:
            text = text.replace("[DEF]", defn).strip(" -")
        # 清理多余空格/逗号
        text = re.sub(r"\s+,", ",", text).strip()
        text = re.sub(r"\(\s*\)", "", text)
        parts.append(text)
    return sep.join(parts) + " ."
