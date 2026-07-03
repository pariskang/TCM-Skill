"""编排:query → 解析 → 多路召回(含语义)→ 加权重排 → LLM 精排 → 模型裁判 → 证据卡片。

流程:
  analyze_query → Recaller.recall(lexical+synonym+graph+semantic) → Reranker.score
  → (可选)rerank_llm 交叉编码器精排 → (逐候选) Extractor → Normalizer →
  EvidenceJudge → Verifier → EvidenceCard → 按等级/相关性排序。
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
    def __init__(self, cfg: EngineConfig, verbose_build=False):
        self.cfg = cfg
        self.onto = load_ontology(cfg.domain)
        self.corpus = Corpus()
        # 每角色可用不同模型;provider=rule 时全部返回 None(走确定性规则)
        self.clients = {r: make_client(cfg.llm.for_role(r)) for r in ROLES}
        # 语义索引(默认 tfidf 离线;可切神经嵌入),带磁盘缓存
        self.semantic_index = self._build_semantic(verbose_build)
        self.reranker = Reranker(self.onto, cfg.weights,
                                 semantic_enabled=self.semantic_index is not None)

    def _build_semantic(self, verbose):
        sem = self.cfg.semantic
        if sem.provider in ("off", "none", ""):
            return None
        from .embeddings import build_index
        embed_llm = None
        if sem.provider != "tfidf":
            # 复用主 LLM 凭据,除非 semantic.llm 单独指定;model 用 semantic.model
            embed_llm = sem.llm or self.cfg.llm.for_role("analyzer")
            if sem.model:
                embed_llm.model = sem.model
        try:
            return build_index(self.corpus.all_passages(), sem,
                               llm_for_embed=embed_llm, verbose=verbose)
        except Exception as e:
            print(f"[graphrag] 语义索引构建失败,降级为无语义:{e}", flush=True)
            return None

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

        # 语义检索也须用繁体查询(语料为繁体),否则简体查询召不回
        from .zh import to_traditional
        sem_query = to_traditional(query)

        recaller = Recaller(self.corpus, self.onto, self.semantic_index)
        cands = recaller.recall(
            analysis["ancient_terms"], analysis["seed_patterns"],
            per_term=max(8, self.cfg.topk_recall // 4), book=book,
            semantic_text=sem_query, semantic_topk=self.cfg.semantic.topk)
        sem_on = "semantic" if self.semantic_index is not None else "semantic 未启用"
        n_sem = sum(1 for c in cands if "semantic" in c.routes)
        log(f"召回 {len(cands)} 条候选(lexical+synonym+graph+{sem_on};其中语义命中 {n_sem})")
        if not cands:
            return {"query_analysis": analysis, "cards": [], "note": "多路召回均无命中"}

        book_dyn = {}
        for c in cands:
            if c.book not in book_dyn:
                book_dyn[c.book] = self.corpus.book_meta(c.book).get("dynasty", "")
        # 语义子分:对全部候选算查询-原文余弦(不止语义路由命中的)
        sem_scores = {}
        if self.semantic_index is not None:
            sem_scores = self.semantic_index.scores(sem_query, [c.passage_id for c in cands])
        cands = self.reranker.score(cands, book_dyn, sem_scores)[: self.cfg.topk_recall]

        # LLM 交叉编码器精排(可选;provider=rule 或关闭时跳过,保持确定性排序)
        if self.cfg.llm_rerank and self.clients.get("reranker") is not None:
            topn = min(self.cfg.llm_rerank_topn, len(cands))
            payload = [{"id": c.passage_id, "cite": c.cite(), "text": c.text}
                       for c in cands[:topn]]
            rel = agents.rerank_llm(self.clients["reranker"], query, payload)
            if rel:
                for c in cands[:topn]:
                    if c.passage_id in rel:
                        # 融合:确定性加权分 0.5 + LLM 精排 0.5,兼顾可复现与语义判断
                        c.subscores["llm_rerank"] = round(rel[c.passage_id], 3)
                        c.score = round(0.5 * c.score + 0.5 * rel[c.passage_id], 4)
                cands.sort(key=lambda x: x.score, reverse=True)
                log(f"LLM 精排完成,重排 top {topn}")
        log(f"进入模型裁判 top {min(self.cfg.topk_cards, len(cands))}")

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
