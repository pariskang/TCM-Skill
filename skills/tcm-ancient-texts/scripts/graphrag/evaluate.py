"""金标准回归评测:验证系统"始终走检索而非记忆"且行为不随改动漂移。

方法学:RAG 系统的可信度必须用固定金标准回归验证(RAGAS 等评测框架的
core practice)。本模块用 provider=rule + tfidf 的**全确定性路径**跑金标准,
任何检索/重排/裁判逻辑的回归都会立即暴露,适合 CI。

用例三类(eval/gold.jsonl,每行一条,均经 tcm.py 实检验证后录入):
  search    — 检索地基:期望书目/段落必须出现在 top-k
  ask       — GraphRAG 端到端:期望条文须入证据卡,可约束最低等级;
              所有卡的 evidence_span 强制通过引用忠实度核验(原文子串)
  exclusion — 排除机制:含禁忌片段(如危候"期六月死")的卡只允许指定等级

指标:
  Recall@k — 期望项在 top-k 中的覆盖率(检索质量主指标)
  MRR      — 首个期望项名次的倒数均值(排序质量;单期望项时与 nDCG 序同)
  排除正确率 / span 忠实度违例数 — 证据安全性指标
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from .agents import span_grounded

DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "eval" / "gold.jsonl"
_GRADE_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}


def load_gold(path: Optional[str] = None) -> List[dict]:
    fp = Path(path) if path else DEFAULT_GOLD
    if not fp.is_file():
        raise FileNotFoundError(f"金标准文件不存在:{fp}")
    cases = []
    for ln in fp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            cases.append(json.loads(ln))
    return cases


def _eval_search(corpus, case: dict, k: int) -> dict:
    rows = corpus.fts_search(case["query"], k, case.get("book"))
    hit_books = [r["book"] for r in rows]
    hit_pass = [(r["book"], r["seq"]) for r in rows]
    expected, found_ranks = [], []
    for b in case.get("expect_books", []):
        expected.append(f"书:{b}")
        rank = next((i + 1 for i, hb in enumerate(hit_books) if hb == b), None)
        found_ranks.append(rank)
    for p in case.get("expect_passages", []):
        expected.append(f"{p['book']}:{p['seq']}")
        rank = next((i + 1 for i, hp in enumerate(hit_pass)
                     if hp == (p["book"], p["seq"])), None)
        found_ranks.append(rank)
    hits = [r for r in found_ranks if r is not None]
    recall = len(hits) / len(expected) if expected else 1.0
    mrr = 1.0 / min(hits) if hits else 0.0
    missing = [e for e, r in zip(expected, found_ranks) if r is None]
    ok = recall >= case.get("min_recall", 1.0)
    return {"recall": recall, "mrr": mrr, "ok": ok,
            "detail": ("全部命中" if not missing else "缺失:" + "、".join(missing))}


def _eval_ask(result: dict, case: dict, k: int) -> dict:
    cards = result["cards"][:k]
    card_keys = [(c.book, c.seq) for c in cards]
    expected, found_ranks, grade_fail = [], [], []
    for p in case.get("expect_passages", []):
        key = f"{p['book']}:{p['seq']}"
        expected.append(key)
        rank = next((i + 1 for i, ck in enumerate(card_keys)
                     if ck == (p["book"], p["seq"])), None)
        found_ranks.append(rank)
        min_g = (case.get("min_grade") or {}).get(key)
        if rank is not None and min_g:
            got = cards[rank - 1].grade
            if _GRADE_RANK.get(got, -1) < _GRADE_RANK.get(min_g, 99):
                grade_fail.append(f"{key} 等级{got}<{min_g}")
    hits = [r for r in found_ranks if r is not None]
    recall = len(hits) / len(expected) if expected else 1.0
    mrr = 1.0 / min(hits) if hits else (1.0 if not expected else 0.0)
    # 引用忠实度:每张卡的 evidence_span 必须是原文子串(确定性核验)
    span_bad = [c.cite() if hasattr(c, "cite") else c.citation for c in cards
                if span_grounded(c.evidence_span, c.original_text) is False]
    n_min = case.get("min_cards", 0)
    missing = [e for e, r in zip(expected, found_ranks) if r is None]
    ok = (recall >= case.get("min_recall", 1.0) and not grade_fail
          and not span_bad and len(cards) >= n_min)
    parts = []
    if missing:
        parts.append("缺失:" + "、".join(missing))
    parts.extend(grade_fail)
    if span_bad:
        parts.append(f"span忠实度违例{len(span_bad)}张")
    if len(cards) < n_min:
        parts.append(f"卡数{len(cards)}<{n_min}")
    return {"recall": recall, "mrr": mrr, "ok": ok,
            "detail": ("通过" if ok else ";".join(parts) or "未通过")}


def _eval_exclusion(result: dict, case: dict) -> dict:
    forbid = case["forbid_substring"]
    allowed = set(case.get("allowed_grades", ["E"]))
    bad = [f"《{c.book}》段{c.seq}={c.grade}" for c in result["cards"]
           if forbid in c.original_text and c.grade not in allowed]
    ok = not bad
    return {"recall": 1.0 if ok else 0.0, "mrr": 1.0 if ok else 0.0, "ok": ok,
            "detail": ("危候均被正确排除" if ok else "违例:" + "、".join(bad))}


def run_eval(cfg, gold_path: Optional[str] = None, k: int = 10,
             out=sys.stdout) -> bool:
    cases = load_gold(gold_path)
    from .corpus import Corpus
    corpus = Corpus()
    engine = None
    ask_cache: dict = {}
    rows, n_ok = [], 0
    try:
        for case in cases:
            t = case["type"]
            if t == "search":
                r = _eval_search(corpus, case, k)
            else:
                if engine is None:
                    from .pipeline import GraphRAG
                    engine = GraphRAG(cfg)
                q = case["query"]
                if q not in ask_cache:
                    ask_cache[q] = engine.ask(q)
                r = (_eval_ask(ask_cache[q], case, k) if t == "ask"
                     else _eval_exclusion(ask_cache[q], case))
            n_ok += int(r["ok"])
            rows.append((case["id"], t, r))
    finally:
        corpus.close()
        if engine is not None:
            engine.close()

    print(f"# 金标准回归评测(共 {len(rows)} 例,k={k},provider={cfg.llm.provider})\n",
          file=out)
    print("| 用例 | 类型 | Recall@k | MRR | 结果 | 说明 |", file=out)
    print("|---|---|---|---|---|---|", file=out)
    for cid, t, r in rows:
        mark = "✅" if r["ok"] else "❌"
        print(f"| {cid} | {t} | {r['recall']:.2f} | {r['mrr']:.2f} "
              f"| {mark} | {r['detail']} |", file=out)
    rec = [r["recall"] for _, _, r in rows]
    mrr = [r["mrr"] for _, _, r in rows]
    print(f"\n**汇总** — 通过 {n_ok}/{len(rows)};"
          f"平均 Recall@{k} = {sum(rec)/len(rec):.3f};"
          f"平均 MRR = {sum(mrr)/len(mrr):.3f}", file=out)
    return n_ok == len(rows)
