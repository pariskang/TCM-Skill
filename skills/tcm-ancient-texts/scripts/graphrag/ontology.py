"""本体加载与查询:L1-L7 术语、同义扩展、排除标签、古今桥接矩阵、图谱边。

数据文件位于 scripts/ontology/*.<domain>.json,纯 JSON(零依赖)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ONTO_DIR = Path(__file__).resolve().parent.parent / "ontology"
_SENT_SPLIT = re.compile(r"[。！？；\n]")

LAYER_LABELS = {
    "L1_disease": "古病名", "L2_pattern": "证候/病机", "L3_manifestation": "症状/表型",
    "L4_population": "人群锚点", "L5_therapy": "治法", "L6_formula_herb": "方药",
    "L7_outcome": "结局/预后",
}

# 简繁转换集中在 zh 模块(opencc 可选 + 内置表 + LLM),此处再导出以兼容旧引用。
from .zh import to_traditional, SIMP2TRAD  # noqa: E402,F401


@dataclass
class Term:
    term: str
    layer: str
    synonyms: List[str]
    exclusion: List[str]


@dataclass
class Ontology:
    domain: str
    label: str
    terms: List[Term] = field(default_factory=list)
    bridges: List[dict] = field(default_factory=list)
    modern_layers: Dict[str, list] = field(default_factory=dict)
    global_exclusions: List[str] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    # 派生索引
    _syn2term: Dict[str, Term] = field(default_factory=dict, repr=False)

    def all_synonyms(self) -> List[str]:
        out = []
        for t in self.terms:
            out.extend(t.synonyms)
        return sorted(set(out), key=len, reverse=True)

    def layer_terms(self, layer: str) -> List[Term]:
        return [t for t in self.terms if t.layer == layer]

    def find_terms_in_text(self, text: str) -> List[Term]:
        """返回文本中出现(任一同义词命中)的术语,去重。"""
        hits = {}
        for t in self.terms:
            for s in t.synonyms:
                if s and s in text:
                    hits[t.term] = t
                    break
        return list(hits.values())

    def exclusions_in_text(self, text: str) -> List[str]:
        found = [e for e in self.global_exclusions if e in text]
        for t in self.terms:
            for e in t.exclusion:
                if e in text and e not in found:
                    found.append(e)
        return found

    def exclusions_scoped(self, text: str) -> dict:
        """句级共现排除判定,替代"全段关键词一票否决"。

        古籍长段常同时含定义性论述与其他语境(如《素问·痿论》既有
        "腎氣熱…發為骨痿"的核心条文,又有"肺熱葉焦"的肺痿论述)。
        全段级排除会把这类定义性条文误杀。判定规则:

        - hard(硬排除):段内出现核心阳性词(L1 病名/L3 表型的同义写法),
          且**每一个**含核心词的句子都同时含排除词——核心词本身处于被排除
          语境(危候预后/外伤/他病),如《玉机真臟论》"大骨枯槁…期六月死"。
        - soft(软排除):排除词在段内出现,但存在至少一个"干净"的核心词句
          (或段内根本无核心词)——上下文噪声,降权即可,不应一票否决。

        返回 {"flags": [...], "hard": bool, "clean_positive": bool}。
        """
        flags = self.exclusions_in_text(text)
        if not flags:
            return {"flags": [], "hard": False, "clean_positive": True}
        core_syns = []
        for t in self.terms:
            if t.layer in ("L1_disease", "L3_manifestation"):
                core_syns.extend(t.synonyms)
        sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        core_sents = [s for s in sents if any(cs in s for cs in core_syns)]
        if not core_sents:
            return {"flags": flags, "hard": False, "clean_positive": False}
        clean = [s for s in core_sents if not any(e in s for e in flags)]
        return {"flags": flags, "hard": not clean, "clean_positive": bool(clean)}

    # -- 图谱路径:从证候节点出发,BFS 到治法/方药/表型 --
    def expand_graph(self, seeds: List[str], max_hops: int = 3) -> Dict[str, list]:
        adj: Dict[str, list] = {}
        for e in self.edges:
            adj.setdefault(e["s"], []).append((e["r"], e["t"]))
        seed_set = set(seeds)
        reached = {}
        frontier = [(s, 0) for s in seeds]
        seen = set(seeds)
        while frontier:
            node, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for rel, tgt in adj.get(node, []):
                if tgt in seed_set:      # 不把种子自身当图谱目标(避免 2-环绕回起点)
                    continue
                reached.setdefault(tgt, []).append((node, rel))
                if tgt not in seen:
                    seen.add(tgt)
                    frontier.append((tgt, hop + 1))
        return reached

    def graph_sources(self) -> set:
        """图谱边的全部源节点(用于确定哪些查询命中词可作图谱种子)。"""
        return {e["s"] for e in self.edges}

    def ppr(self, seeds: List[str], damping: float = 0.85,
            iters: int = 30, tol: float = 1e-8) -> Dict[str, float]:
        """Personalized PageRank:以查询命中词为重启种子,在知识图谱上做
        带重启随机游走,得到每个节点与查询的图结构相关度。

        相比 BFS 逐跳扩展,PPR 把「离种子近、且被多条路径汇聚」的节点排在前面,
        多跳呈几何衰减——这是 HippoRAG(NeurIPS 2024)验证过的图谱召回打分方式。
        医学关系边(manifests/treated_by/uses_herb/relieves)语义上双向可达,
        按无向图游走以保证连通性。图谱规模小(数十边),幂迭代开销可忽略。
        """
        adj: Dict[str, list] = {}
        nodes = set()
        for e in self.edges:
            nodes.add(e["s"]); nodes.add(e["t"])
            adj.setdefault(e["s"], []).append(e["t"])
            adj.setdefault(e["t"], []).append(e["s"])
        seed_in = [s for s in seeds if s in nodes]
        if not seed_in:
            return {}
        restart = {n: (1.0 / len(seed_in) if n in seed_in else 0.0) for n in nodes}
        rank = dict(restart)
        for _ in range(iters):
            nxt = {n: (1 - damping) * restart[n] for n in nodes}
            for n, r in rank.items():
                outs = adj.get(n)
                if not outs:
                    continue
                share = damping * r / len(outs)
                for t in outs:
                    nxt[t] += share
            delta = sum(abs(nxt[n] - rank[n]) for n in nodes)
            rank = nxt
            if delta < tol:
                break
        return rank

    def bridge_phenotypes(self) -> List[str]:
        out = []
        for b in self.bridges:
            out.extend(b.get("modern_phenotype", []))
        return sorted(set(out))


def _load_json(name: str, domain: str) -> Optional[dict]:
    fp = _ONTO_DIR / f"{name}.{domain}.json"
    if not fp.is_file():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def available_domains() -> List[str]:
    doms = set()
    for fp in _ONTO_DIR.glob("terms.*.json"):
        doms.add(fp.stem.split(".", 1)[1])
    return sorted(doms)


def load_ontology(domain: str = "osteoporosis") -> Ontology:
    tdata = _load_json("terms", domain)
    if not tdata:
        raise FileNotFoundError(
            f"未找到本体 terms.{domain}.json;可用病种: {available_domains()}")
    onto = Ontology(domain=domain, label=tdata.get("label", domain))
    for layer, items in tdata.get("terms", {}).items():
        for it in items:
            t = Term(term=it["term"], layer=layer,
                     synonyms=it.get("synonyms") or [it["term"]],
                     exclusion=it.get("exclusion") or [])
            onto.terms.append(t)
            for s in t.synonyms:
                onto._syn2term[s] = t

    bdata = _load_json("bridge", domain) or {}
    onto.bridges = bdata.get("bridges", [])
    onto.modern_layers = bdata.get("modern_layers", {})
    onto.global_exclusions = bdata.get("global_exclusions", [])

    rdata = _load_json("relations", domain) or {}
    onto.edges = rdata.get("edges", [])
    return onto
