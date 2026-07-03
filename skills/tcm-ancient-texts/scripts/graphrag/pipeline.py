"""编排:query → 解析 → 四路召回 → 加权重排 → 五重模型角色 → 证据卡片。

流程:
  analyze_query → Recaller.recall → Reranker.score → (逐候选) Extractor →
  Normalizer → EvidenceJudge → Verifier → EvidenceCard → 按等级/相关性排序。
"""
from __future__ import annotations

from typing import List, Optional

from . import agents
from .config import EngineConfig, ROLES
from .corpus import Corpus
from .evidence import Candidate, EvidenceCard
from .llm import make_client
from .ontology import load_ontology
from .recall import Recaller
from .rerank import Reranker

_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


class GraphRAG:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.onto = load_ontology(cfg.domain)
        self.corpus = Corpus()
        # 每角色可用不同模型;provider=rule 时全部返回 None(走确定性规则)
        self.clients = {r: make_client(cfg.llm.for_role(r)) for r in ROLES}
        self.reranker = Reranker(self.onto, cfg.weights, semantic_enabled=False)

    def close(self):
        self.corpus.close()

    def ask(self, query: str, book: Optional[str] = None, verbose=False) -> dict:
        log = (lambda m: print(f"[graphrag] {m}", flush=True)) if verbose else (lambda m: None)

        analysis = agents.analyze_query(self.clients["analyzer"], query, self.onto)
        log(f"解析:古籍检索词={analysis['ancient_terms']} 证候种子={analysis['seed_patterns']}")

        if not analysis.get("has_domain_signal") and not analysis["ancient_terms"]:
            return {"query_analysis": analysis, "cards": [],
                    "note": (f"查询未匹配到病种「{self.onto.label}」的任何古籍术语、证候或"
                             f"现代表型。请确认查询与当前病种相关(--domain 可切换病种),"
                             f"或改用该病种相关的证候/症状/古籍词。")}

        recaller = Recaller(self.corpus, self.onto)
        cands = recaller.recall(analysis["ancient_terms"], analysis["seed_patterns"],
                                per_term=max(8, self.cfg.topk_recall // 4), book=book)
        log(f"召回 {len(cands)} 条候选(lexical+synonym+graph 合并去重;semantic 未启用)")
        if not cands:
            return {"query_analysis": analysis, "cards": [], "note": "四路召回均无命中"}

        book_dyn = {}
        for c in cands:
            if c.book not in book_dyn:
                book_dyn[c.book] = self.corpus.book_meta(c.book).get("dynasty", "")
        cands = self.reranker.score(cands, book_dyn)[: self.cfg.topk_recall]
        log(f"重排完成,进入模型裁判 top {min(self.cfg.topk_cards, len(cands))}")

        cards: List[EvidenceCard] = []
        for c in cands[: self.cfg.topk_cards]:
            try:
                cards.append(self._build_card(c, analysis))
            except Exception as e:   # 单张卡片(如某次 LLM 返回异常)不拖垮整个 ask
                log(f"跳过候选 段{c.seq}《{c.book}》:{type(e).__name__}: {e}")
        cards = [k for k in cards if k]

        cards.sort(key=lambda k: (_GRADE_ORDER.get(k.grade, 9), -k.relevance_score))
        return {"query_analysis": analysis, "cards": cards,
                "n_candidates": len(cands)}

    def _build_card(self, c: Candidate, analysis: dict) -> EvidenceCard:
        meta = self.corpus.book_meta(c.book)
        # 1. 抽取
        ents = agents.extract(self.clients["extractor"], c.text, self.onto)
        # 2. 归一
        norm = agents.normalize(self.clients["normalizer"], ents, self.onto, c.text)
        excl = getattr(c, "_exclusions", None)
        if excl is None:
            excl = self.onto.exclusions_in_text(c.text)
        card_data = {
            "ancient_disease_terms": ents.get("ancient_disease_terms", []),
            "manifestations": ents.get("manifestations", []),
            "patterns": ents.get("patterns", []),
            "therapeutic_principles": ents.get("therapeutic_principles", []),
            "formulas_or_herbs": ents.get("formulas_or_herbs", []),
            "population": ents.get("population", []),
            "modern_phenotypes": norm.get("modern_phenotypes", []),
            "mapping_type": norm.get("mapping_type", ""),
            "exclusion_flags": excl,
        }
        # 3. 裁判
        verdict = agents.judge(self.clients["judge"], analysis, card_data, self.onto)
        # 4. 核验
        span = ents.get("evidence_span", "")
        vr = agents.verify(self.clients["verifier"], span, card_data, c.text, self.onto)

        # 相关性:模型裁判分与召回重排分融合
        rel = round(0.6 * float(verdict.get("relevance_score", 0)) + 0.4 * c.score, 3)
        return EvidenceCard(
            passage_id=c.passage_id, citation=c.cite(), book=c.book, seq=c.seq,
            path=c.path, original_text=c.text, evidence_span=span,
            modern_translation=ents.get("modern_translation", ""),
            ancient_disease_terms=card_data["ancient_disease_terms"],
            manifestations=card_data["manifestations"],
            patterns=card_data["patterns"],
            therapeutic_principles=card_data["therapeutic_principles"],
            formulas_or_herbs=card_data["formulas_or_herbs"],
            population=card_data["population"],
            modern_phenotypes=card_data["modern_phenotypes"],
            mapping_type=card_data["mapping_type"],
            relevance_score=rel, grade=verdict.get("grade", "C"),
            inclusion_reason=verdict.get("reason", ""),
            exclusion_flags=excl,
            grounded=vr.get("grounded"), verifier_note=vr.get("note", ""),
            dynasty=meta.get("dynasty", ""), author=meta.get("author", ""),
            edition=meta.get("edition", ""), quality=meta.get("quality", ""),
            routes=c.routes,
        )
