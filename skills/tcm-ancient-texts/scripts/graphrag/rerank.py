"""加权重排:综合评分而非让模型凭感觉。

S_final = 0.20·lexical + 0.20·semantic + 0.15·ontology + 0.15·phenotype
        + 0.10·context + 0.10·evidence + 0.05·dynasty - 0.05·exclusion

无 semantic 分(未启用向量)时,自动把其权重按比例分摊给 lexical/ontology,
保证分值可比。所有子分归一到 [0,1]。
"""
from __future__ import annotations

from typing import Dict, List

from .evidence import Candidate
from .ontology import Ontology

DEFAULT_WEIGHTS = {
    "lexical": 0.20, "semantic": 0.20, "ontology": 0.15, "phenotype": 0.15,
    "context": 0.10, "evidence": 0.10, "dynasty": 0.05, "exclusion": 0.05,
}

# 朝代/文献权威性先验(粗粒度,可在 bridge 中细化)
_DYNASTY_WEIGHT = {
    "戰國": 1.0, "戰國至西漢": 1.0, "秦漢": 0.95, "東漢": 0.95, "漢": 0.9,
    "晉": 0.85, "隋": 0.8, "唐": 0.85, "宋": 0.8, "金": 0.75, "元": 0.75,
    "明": 0.7, "清": 0.65, "民國": 0.55,
}
_CHAIN_LAYERS = ("L1_disease", "L2_pattern", "L3_manifestation",
                 "L5_therapy", "L6_formula_herb")


class Reranker:
    def __init__(self, onto: Ontology, weights: Dict[str, float] = None,
                 semantic_enabled: bool = False):
        self.onto = onto
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.semantic_enabled = semantic_enabled
        # 从本体 L1(病名)/L3(表型)术语字符派生 context 加成关键词,
        # 避免写死某病种字符(换病种后仍能正确加成)。
        self._context_chars = set()
        for t in onto.terms:
            if t.layer in ("L1_disease", "L3_manifestation"):
                self._context_chars.update(t.term)
        # 若无语义分,把 semantic 权重分摊给 lexical+ontology
        if not semantic_enabled:
            s = self.w.pop("semantic", 0.0)
            self.w["lexical"] += s * 0.6
            self.w["ontology"] += s * 0.4

    def _layers_present(self, text: str) -> set:
        layers = set()
        for t in self.onto.terms:
            if any(syn in text for syn in t.synonyms):
                layers.add(t.layer)
        return layers

    def score(self, cands: List[Candidate], book_dynasty: Dict[str, str],
              semantic_scores: Dict[int, float] = None) -> List[Candidate]:
        semantic_scores = semantic_scores or {}
        max_terms = max((len(c.matched_terms) for c in cands), default=1) or 1
        for c in cands:
            text = c.text
            sub = {}
            # lexical:命中词数 + 是否有 FTS 路由
            sub["lexical"] = min(1.0, len(c.matched_terms) / max_terms
                                 + (0.2 if "lexical" in c.routes else 0))
            # semantic(钳位到 [0,1],防外部向量分越界破坏可比性)
            sub["semantic"] = min(1.0, max(0.0, semantic_scores.get(c.passage_id, 0.0)))
            # ontology:命中的 L 层覆盖度
            layers = self._layers_present(text)
            sub["ontology"] = len(layers & set(_CHAIN_LAYERS)) / len(_CHAIN_LAYERS)
            # phenotype:桥接矩阵现代表型覆盖(经古籍词反查)
            pheno_hit = self._phenotype_coverage(text)
            sub["phenotype"] = pheno_hit
            # context:证据链完整性的上下文信号(标题路径含相关字 + 长度适中)
            plen = len(text)
            sub["context"] = (0.5 if 80 <= plen <= 600 else 0.2 if plen < 80 else 0.35)
            path_bonus = 0.5 if any(ch in self._context_chars for ch in c.path) else 0.0
            sub["context"] = min(1.0, sub["context"] + path_bonus)
            # evidence:是否形成 病名/表型 + 病机 + 治法/方药 链
            has_symptom = layers & {"L1_disease", "L3_manifestation"}
            has_pattern = "L2_pattern" in layers
            has_treat = layers & {"L5_therapy", "L6_formula_herb"}
            sub["evidence"] = (int(bool(has_symptom)) + int(has_pattern)
                               + int(bool(has_treat))) / 3.0
            # dynasty
            dyn = book_dynasty.get(c.book, "")
            sub["dynasty"] = next((w for k, w in _DYNASTY_WEIGHT.items()
                                   if k and k in dyn), 0.5)
            # exclusion(惩罚项)
            excl = self.onto.exclusions_in_text(text)
            sub["exclusion"] = min(1.0, len(excl) / 2.0)
            c._exclusions = excl  # 缓存供裁判使用

            total = 0.0
            for k, wv in self.w.items():
                if k == "exclusion":
                    total -= wv * sub.get(k, 0.0)
                else:
                    total += wv * sub.get(k, 0.0)
            c.subscores = {k: round(v, 3) for k, v in sub.items()}
            c.score = round(max(0.0, total), 4)
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def _phenotype_coverage(self, text: str) -> float:
        if not self.onto.bridges:
            return 0.0
        hit = 0
        for b in self.onto.bridges:
            if any(a in text for a in b.get("ancient", [])):
                hit += 1
        return min(1.0, hit / max(1, len(self.onto.bridges)))
