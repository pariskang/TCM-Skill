"""多路召回:lexical(FTS5/BM25)+ synonym(本体同义扩展)+ graph(PPR 图谱路径)
+ semantic(向量语义),合并去重为候选证据集。

方法学要点:
- 每条召回列表(route:term 一次查询为一个列表)内的名次被完整记录到
  Candidate.route_ranks,供重排层做 Reciprocal Rank Fusion(RRF,
  Cormack et al., SIGIR 2009)——只用名次不用分值,免疫 BM25 rank/
  余弦/LIKE 密度代理三种分值尺度不可比的问题。
- 图谱路由用 Personalized PageRank(HippoRAG, NeurIPS 2024 的图谱召回
  打分方式)对扩展目标排序:离种子近且被多条路径汇聚的节点优先检索,
  多跳几何衰减,替代原先"BFS 逐跳一视同仁"。
- FTS5 的 rank 列即内建 BM25(Robertson & Zaragoza 2009),lexical 列表
  本身已按 BM25 排序,其名次进入 RRF 后 BM25 信号自然保留。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .corpus import Corpus
from .evidence import Candidate
from .ontology import Ontology


def _merge(pool: Dict[int, Candidate], rows, route: str, term: str,
           graph_trace: Optional[List[str]] = None):
    """把一条召回列表并入候选池,并记录列表内名次(1 起,保留最好名次)。"""
    list_key = f"{route}:{term or '~'}"
    for pos, r in enumerate(rows, 1):
        c = pool.get(r["id"])
        if not c:
            c = Candidate(passage_id=r["id"], book=r["book"], seq=r["seq"],
                          path=r.get("path", ""), text=r["text"])
            pool[r["id"]] = c
        if route not in c.routes:
            c.routes.append(route)
        if term and term not in c.matched_terms:
            c.matched_terms.append(term)
        prev = c.route_ranks.get(list_key)
        if prev is None or pos < prev:
            c.route_ranks[list_key] = pos
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
            _merge(pool, self.corpus.fts_search(qt, per_term, book), "lexical", qt)
            seen_terms.add(qt)
            # 同义扩展
            for t in self.onto.terms:
                if qt in t.synonyms or qt == t.term:
                    for syn in t.synonyms:
                        if syn in seen_terms:
                            continue
                        seen_terms.add(syn)
                        _merge(pool, self.corpus.fts_search(syn, per_term // 2, book),
                               "synonym", syn)

        # 路由 3:图谱路径召回 —— PPR 对扩展目标打分排序,
        # 优先检索"离证候种子近、多路径汇聚"的表型/治法/方药节点。
        if seed_patterns:
            reached = self.onto.expand_graph(seed_patterns)
            ppr = self.onto.ppr(seed_patterns)
            targets = sorted(reached, key=lambda t: ppr.get(t, 0.0), reverse=True)
            for tgt in targets[:8]:                      # PPR top-8 目标
                provenance = reached[tgt]
                trace = [f"{s}—{r}→{tgt}" for s, r in provenance[:2]]
                # 目标节点可能是术语,取其同义词检索
                terms_to_search = [tgt]
                for t in self.onto.terms:
                    if tgt == t.term or tgt in t.synonyms:
                        # 含规范名本身(个别术语规范名未收进 synonyms)
                        terms_to_search = list(dict.fromkeys([t.term] + t.synonyms))
                        break
                for st in terms_to_search[:3]:
                    _merge(pool, self.corpus.fts_search(st, per_term // 3, book),
                           "graph", st, graph_trace=trace)

        # 路由 4:语义召回 —— 查询向量最近邻,补齐"用词相近但未精确命中"的条文
        if self.semantic_index is not None and semantic_text:
            hits = self.semantic_index.query(semantic_text, topk=semantic_topk)
            need = [pid for pid, _ in hits if pid not in pool]
            fetched = self.corpus.passages_by_ids(need)
            for pos, (pid, sc) in enumerate(hits, 1):
                row = pool.get(pid)
                if row is None:
                    src = fetched.get(pid)
                    if not src or (book and src["book"] != book):
                        continue
                    _merge(pool, [src], "semantic", "")
                    pool[pid].route_ranks["semantic:~"] = pos
                else:
                    if "semantic" not in row.routes:
                        row.routes.append("semantic")
                    prev = row.route_ranks.get("semantic:~")
                    if prev is None or pos < prev:
                        row.route_ranks["semantic:~"] = pos

        return list(pool.values())
