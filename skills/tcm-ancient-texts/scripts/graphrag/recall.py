"""多路召回:lexical(BM25/FTS)+ synonym(本体同义扩展)+ graph(图谱路径扩展),
合并去重为候选证据集。

设计上预留第四路 semantic(向量语义),为可选增强:仅当配置了 embedding provider
时启用;当前实现默认不启用,也不伪造语义分(rerank 中 semantic 权重按比例分摊)。
这符合"确定性地基优先、概率性增强其次"的设计原则;接入方式见 docs/GRAPHRAG.md 路线图。
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
    def __init__(self, corpus: Corpus, onto: Ontology, semantic_index=None):
        self.corpus = corpus
        self.onto = onto
        self.semantic_index = semantic_index

    def recall(self, query_terms: List[str], seed_patterns: List[str],
               per_term: int = 20, book: Optional[str] = None,
               semantic_text: Optional[str] = None,
               semantic_topk: int = 40) -> List[Candidate]:
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
                    # 含规范名本身(个别术语规范名未收进 synonyms)
                    terms_to_search = list(dict.fromkeys([t.term] + t.synonyms))
                    break
            for st in terms_to_search[:3]:
                for r in self.corpus.fts_search(st, per_term // 3, book):
                    _merge(pool, [r], "graph", st, graph_trace=trace)

        # 路由 3:语义召回 —— 查询向量最近邻,补齐"用词相近但未精确命中"的条文
        if self.semantic_index is not None and semantic_text:
            hits = self.semantic_index.query(semantic_text, topk=semantic_topk)
            need = [pid for pid, _ in hits if pid not in pool]
            fetched = self.corpus.passages_by_ids(need)
            for pid, sc in hits:
                row = pool.get(pid)
                if row is None:
                    src = fetched.get(pid)
                    if not src or (book and src["book"] != book):
                        continue
                    _merge(pool, [src], "semantic", "")
                elif "semantic" not in row.routes:
                    row.routes.append("semantic")

        return list(pool.values())
