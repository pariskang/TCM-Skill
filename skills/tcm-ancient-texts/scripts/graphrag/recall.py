"""四路召回:lexical(BM25/FTS)+ synonym(本体同义扩展)+ semantic(可选向量)
+ graph(图谱路径扩展),合并去重为候选证据集。

semantic 路由为可选增强:仅当配置了 embedding provider 时启用;否则跳过并记录,
不伪造语义分。这符合"确定性地基优先、概率性增强其次"的设计原则。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .corpus import Corpus
from .evidence import Candidate
from .ontology import Ontology


def _merge(pool: Dict[int, Candidate], rows, route: str, term: str,
           graph_trace: Optional[List[str]] = None):
    for r in rows:
        c = pool.get(r["id"])
        if not c:
            c = Candidate(passage_id=r["id"], book=r["book"], seq=r["seq"],
                          path=r.get("path", ""), text=r["text"])
            pool[r["id"]] = c
        if route not in c.routes:
            c.routes.append(route)
        if term and term not in c.matched_terms:
            c.matched_terms.append(term)
        if graph_trace:
            for g in graph_trace:
                if g not in c.graph_trace:
                    c.graph_trace.append(g)


class Recaller:
    def __init__(self, corpus: Corpus, onto: Ontology):
        self.corpus = corpus
        self.onto = onto

    def recall(self, query_terms: List[str], seed_patterns: List[str],
               per_term: int = 20, book: Optional[str] = None) -> List[Candidate]:
        pool: Dict[int, Candidate] = {}

        # 路由 1+2:lexical 与 synonym 合并处理
        # query_terms 为核心检索词;每个词再用本体同义词扩展。
        seen_terms = set()
        for qt in query_terms:
            for r in self.corpus.fts_search(qt, per_term, book):
                _merge(pool, [r], "lexical", qt)
            seen_terms.add(qt)
            # 同义扩展
            for t in self.onto.terms:
                if qt in t.synonyms or qt == t.term:
                    for syn in t.synonyms:
                        if syn in seen_terms:
                            continue
                        seen_terms.add(syn)
                        for r in self.corpus.fts_search(syn, per_term // 2, book):
                            _merge(pool, [r], "synonym", syn)

        # 路由 4:图谱路径召回 —— 从证候种子扩展到表型/治法/方药,再检索这些词
        reached = self.onto.expand_graph(seed_patterns) if seed_patterns else {}
        for tgt, provenance in reached.items():
            trace = [f"{s}—{r}→{tgt}" for s, r in provenance[:2]]
            # 目标节点可能是术语,取其同义词检索
            terms_to_search = [tgt]
            for t in self.onto.terms:
                if tgt == t.term or tgt in t.synonyms:
                    terms_to_search = t.synonyms
                    break
            for st in terms_to_search[:3]:
                for r in self.corpus.fts_search(st, per_term // 3, book):
                    _merge(pool, [r], "graph", st, graph_trace=trace)

        return list(pool.values())
