"""本体加载与查询:L1-L7 术语、同义扩展、排除标签、古今桥接矩阵、图谱边。

数据文件位于 scripts/ontology/*.<domain>.json,纯 JSON(零依赖)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ONTO_DIR = Path(__file__).resolve().parent.parent / "ontology"

LAYER_LABELS = {
    "L1_disease": "古病名", "L2_pattern": "证候/病机", "L3_manifestation": "症状/表型",
    "L4_population": "人群锚点", "L5_therapy": "治法", "L6_formula_herb": "方药",
    "L7_outcome": "结局/预后",
}

# 最小简→繁映射:覆盖本体术语及常见中医字,让离线 rule 路径不因简体查询丢词。
# 非完整转换(完整转换交给 LLM analyzer 或 opencc);仅保证常见查询可用。
SIMP2TRAD = {
    "软": "軟", "肾": "腎", "阴": "陰", "阳": "陽", "证": "證", "脉": "脈",
    "气": "氣", "热": "熱", "湿": "濕", "风": "風", "经": "經", "络": "絡",
    "脏": "臟", "虚": "虛", "补": "補", "泻": "瀉", "归": "歸", "术": "朮",
    "断": "斷", "续": "續", "龟": "龜", "摇": "搖", "齿": "齒", "艰": "艱",
    "妇": "婦", "丝": "絲", "医": "醫", "药": "藥", "针": "針", "壮": "壯",
    "肠": "腸", "节": "節", "涩": "澀", "无": "無", "后": "後", "发": "發",
    "会": "會", "体": "體", "灵": "靈", "卫": "衛", "营": "營", "痹": "痹",
    "带": "帶", "痉": "痙", "厥": "厥", "疸": "疸", "痫": "癇", "瘅": "癉",
    "浊": "濁", "涌": "湧", "渗": "滲", "颤": "顫", "举": "舉", "废": "廢",
}


def to_traditional(text: str) -> str:
    """按最小映射把简体字逐字转繁体(未收录字原样保留)。"""
    return "".join(SIMP2TRAD.get(c, c) for c in text)


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
