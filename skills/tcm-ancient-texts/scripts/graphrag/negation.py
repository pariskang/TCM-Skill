"""方剂/治法极性检测:区分"处方(该用)"与"禁忌(禁用)"。

古籍(尤其《伤寒论》)对同一方剂在不同证候下立场相反:
  桂枝湯主之        → 处方(affirm)
  不可與桂枝湯       → 禁忌(negate)
  桂枝不中與之也     → 禁忌(negate)
  可與大承氣湯…不可與之 → 条件辨证(conditional,同段肯定+否定)
朴素检索把这几类当同等相关,是证据系统的科学错误。

本模块是临床 NLP 否定检测经典算法(NegEx, Chapman et al. 2001;ConText,
Harkema et al. 2009)对文言中医文本的适配:极性是**每个方剂目标**的属性,
由**线索词 + 分句作用域**决定——一段可同时肯定 A 方、否定 B 方。零依赖。

用法:
    from graphrag.negation import classify, passage_polarity
    classify("若酒客病，不可與桂枝湯", ["桂枝湯"])
      → {"桂枝湯": {"polarity": "negate", "cues": ["不可與"]}}
"""
from __future__ import annotations

import re
from typing import Dict, List

_CLAUSE_SPLIT = re.compile(r"[，。；！？、：\n]")

# 否定/禁忌线索(前置:线索在方剂名之前)。长词优先,避免"不可"吃掉"不可與"。
_NEG_PRE = ("不可與", "不可服", "不可用", "未可與", "勿與", "勿服", "慎不可與",
            "慎不可", "不宜與", "不宜服", "不宜用", "不宜", "禁用", "切禁", "忌服")
# 否定后置线索(方剂名在前:X不中與之 / X不可與之)
_NEG_POST = ("不中與之", "不中與", "不可與之", "不可用之")
# 全分句级否定代词线索(之=前文主方):不可與之 / 不中與之 单独成句时回指主方
_NEG_ANAPHOR = ("不可與之", "不中與之", "不可用之")
# 肯定后置(方剂名 + 主之/則愈…)
_AFF_POST = ("主之", "則愈", "則解", "則差", "則已")
# 肯定前置(宜/與/可與 + 方剂名)。检查时须确认其前不含否定字。
_AFF_PRE = ("可與", "當與", "先與", "卻與", "更與", "復與", "宜", "屬", "與")
_NEG_CHARS = ("不", "勿", "未", "禁", "忌", "慎")
# 危险警示(方剂 + 下咽 … 斃/亡/死)
_HARM = ("下咽",)
_HARM_OUTCOME = ("斃", "亡", "死")

POLARITIES = ("affirm", "negate", "conditional", "neutral")


def _occurrence_polarity(clause: str, tgt: str, start: int) -> str:
    """判定 clause 中位于 start 处的一次方剂出现的极性。"""
    before = clause[:start]
    after = clause[start + len(tgt):]

    # 1) 后置否定:X不中與 / X不可與之(紧跟或隔"之""也")
    tail = after[:4]
    if any(tail.startswith(c) or tail.startswith("之" + c) for c in _NEG_POST):
        return "negate"
    # 2) 危险警示:X下咽 … 斃/亡/死
    if any(after.startswith(h) for h in _HARM) and any(o in after for o in _HARM_OUTCOME):
        return "negate"
    # 3) 前置否定:方剂名紧接在否定线索之后(不可與桂枝湯)
    for cue in _NEG_PRE:
        if before.endswith(cue):
            return "negate"
    # 4) 前置肯定:宜/與/可與 + 方剂名,且线索前无否定字
    for cue in _AFF_PRE:
        if before.endswith(cue):
            pre = before[: len(before) - len(cue)]
            if pre and pre[-1] in _NEG_CHARS:      # 不可與/未可與… 已在上面处理
                continue
            return "affirm"
    # 5) 后置肯定:X … 主之/則愈(同分句内)
    if any(a in after for a in _AFF_POST):
        return "affirm"
    return "neutral"


def classify(text: str, targets: List[str]) -> Dict[str, dict]:
    """对每个方剂目标返回 {"polarity", "cues", "occurrences"}。

    polarity 聚合规则:全否定→negate;全肯定→affirm;既肯定又否定→conditional
    (辨证使用);仅中性提及→neutral。
    """
    clauses = []
    pos = 0
    for m in _CLAUSE_SPLIT.finditer(text):
        clauses.append((pos, text[pos:m.start()]))
        pos = m.end()
    if pos < len(text):
        clauses.append((pos, text[pos:]))

    out: Dict[str, dict] = {}
    # 段级回指否定(不可與之/不中與之 独立成句):作用于"主方"——
    # 取在文中最早出现且无自身肯定线索的目标。先扫描是否存在此类线索。
    anaphor_clauses = [c for _, c in clauses
                       if any(c.strip() == a or c.strip() == a + "也" for a in _NEG_ANAPHOR)]

    for tgt in targets:
        if not tgt:
            continue
        labels: List[str] = []
        cues: List[str] = []
        occ = 0
        for _, clause in clauses:
            idx = clause.find(tgt)
            while idx != -1:
                occ += 1
                pol = _occurrence_polarity(clause, tgt, idx)
                labels.append(pol)
                if pol == "negate":     # 只记真正作用于该目标出现的否定线索
                    for cue in _NEG_PRE + _NEG_POST:
                        if cue in clause and cue not in cues:
                            cues.append(cue)
                idx = clause.find(tgt, idx + len(tgt))
        # 回指否定:若该目标是文中首个方剂且存在"不可與之"独立分句,补记否定
        if anaphor_clauses and tgt in text:
            first_pos = min((text.find(t) for t in targets if t and t in text),
                            default=-1)
            if text.find(tgt) == first_pos:
                labels.append("negate")
                for a in _NEG_ANAPHOR:
                    if any(a in c for c in anaphor_clauses) and a not in cues:
                        cues.append(a)
        if not labels:
            out[tgt] = {"polarity": "neutral", "cues": [], "occurrences": 0}
            continue
        has_aff = "affirm" in labels
        has_neg = "negate" in labels
        if has_aff and has_neg:
            polarity = "conditional"
        elif has_neg:
            polarity = "negate"
        elif has_aff:
            polarity = "affirm"
        else:
            polarity = "neutral"
        out[tgt] = {"polarity": polarity, "cues": cues, "occurrences": occ}
    return out


def passage_polarity(text: str, targets: List[str]) -> str:
    """整段对给定方剂集合的聚合立场:任一 conditional→conditional;
    有肯定有否定→conditional;全否定→negate;有肯定→affirm;否则 neutral。"""
    labels = {v["polarity"] for v in classify(text, targets).values()}
    labels.discard("neutral")
    if not labels:
        return "neutral"
    if labels == {"affirm"}:
        return "affirm"
    if labels == {"negate"}:
        return "negate"
    return "conditional"


_POLARITY_LABEL = {
    "affirm": "处方(该证可用)", "negate": "禁忌(此证禁用)",
    "conditional": "辨证使用(随证可用可禁)", "neutral": "论述(非直接方证)",
}


def polarity_label(polarity: str) -> str:
    return _POLARITY_LABEL.get(polarity, polarity)
