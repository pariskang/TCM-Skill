"""语义召回后端 —— 两条真实可用的实现,统一接口。

SemanticIndex 接口:
  query(text, topk)     -> [(passage_id, score∈[0,1]), ...]   语义召回
  scores(text, pids)    -> {passage_id: score∈[0,1]}          给 rerank 的语义子分

两种后端:
  TfidfIndex   离线、零依赖:字符 n-gram(2+3)TF-IDF + 倒排索引 + 余弦。
               统计向量空间,能召回"用词相近但非精确子串"的条文;确定性、可复现、
               可缓存;38k 段秒级。默认后端。
  DenseIndex   神经嵌入(litellm / openai / azure),语义级;向量磁盘缓存。
               需对应 SDK;大规模建议装 numpy 加速余弦。

磁盘缓存在 data/ 下,按 (后端, 模型, 段落数) 指纹命名,重建即失效。
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import tcm  # noqa: E402


def _cache_dir() -> Path:
    d = tcm.data_dir() / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 字符 n-gram 特征
# ---------------------------------------------------------------------------

def _ngrams(text: str, ns=(2, 3)) -> List[str]:
    # 只保留 CJK 与字母数字,去标点空白
    chars = [c for c in text if c.strip() and not _is_punct(c)]
    s = "".join(chars)
    grams = []
    for n in ns:
        if len(s) >= n:
            grams.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return grams


_PUNCT = set("，。、；：！？「」『』（）()《》〈〉·…—　,.;:!?\"'[]{}<>/\\|@#$%^&*-_=+~`")


def _is_punct(c: str) -> bool:
    return c in _PUNCT


# ---------------------------------------------------------------------------
# 离线:字符 n-gram TF-IDF 倒排索引
# ---------------------------------------------------------------------------

class TfidfIndex:
    NAME = "tfidf"
    TOP_FEATURES = 160   # 每段保留权重最高的特征数,限内存/加速

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, float]]] = {}
        self.n_docs = 0

    def _vectorize(self, text: str) -> Dict[str, float]:
        grams = _ngrams(text)
        if not grams:
            return {}
        tf: Dict[str, int] = {}
        for g in grams:
            tf[g] = tf.get(g, 0) + 1
        vec = {}
        for g, c in tf.items():
            idf = self.idf.get(g)
            if idf is None:
                continue
            vec[g] = (1.0 + math.log(c)) * idf
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {g: w / norm for g, w in vec.items()}

    def build(self, corpus: List[Tuple[int, str]], verbose=False):
        # 1) DF
        df: Dict[str, int] = {}
        for _, text in corpus:
            for g in set(_ngrams(text)):
                df[g] = df.get(g, 0) + 1
        self.n_docs = len(corpus)
        # 稀有特征(df<2)与超高频(df>60% 文档)剔除,降噪+省内存
        hi = max(3, int(self.n_docs * 0.6))
        self.idf = {g: math.log(self.n_docs / (1 + d))
                    for g, d in df.items() if 2 <= d <= hi}
        # 2) 倒排索引(每段截取 top 特征)
        self.postings = {}
        for pid, text in corpus:
            vec = self._vectorize(text)
            if not vec:
                continue
            top = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)[: self.TOP_FEATURES]
            for g, w in top:
                self.postings.setdefault(g, []).append((pid, w))
        if verbose:
            print(f"[semantic] tfidf 索引:{self.n_docs} 段,{len(self.idf)} 特征",
                  file=sys.stderr)

    def query(self, text: str, topk: int = 50) -> List[Tuple[int, float]]:
        qv = self._vectorize(text)
        if not qv:
            return []
        scores: Dict[int, float] = {}
        for g, qw in qv.items():
            for pid, pw in self.postings.get(g, ()):    # 已归一,点积即余弦
                scores[pid] = scores.get(pid, 0.0) + qw * pw
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        return [(pid, min(1.0, s)) for pid, s in ranked]

    def scores(self, text: str, pids: List[int]) -> Dict[int, float]:
        qv = self._vectorize(text)
        if not qv:
            return {}
        want = set(pids)
        acc: Dict[int, float] = {}
        for g, qw in qv.items():
            for pid, pw in self.postings.get(g, ()):
                if pid in want:
                    acc[pid] = acc.get(pid, 0.0) + qw * pw
        return {pid: min(1.0, s) for pid, s in acc.items()}

    # -- 持久化 --
    def to_blob(self) -> dict:
        return {"idf": self.idf, "postings": self.postings, "n_docs": self.n_docs}

    def from_blob(self, blob: dict):
        self.idf = blob["idf"]
        self.postings = blob["postings"]
        self.n_docs = blob["n_docs"]


# ---------------------------------------------------------------------------
# 神经嵌入:embedder(懒加载 SDK)+ DenseIndex
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, cfg):
        self.cfg = cfg   # LLMConfig(复用同一凭据体系)

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class LiteLLMEmbedder(Embedder):
    def embed(self, texts):
        import litellm
        kw = dict(model=self.cfg.model, input=texts)
        if self.cfg.api_key:
            kw["api_key"] = self.cfg.api_key
        if self.cfg.api_base:
            kw["api_base"] = self.cfg.api_base
        r = litellm.embedding(**kw)
        return [d["embedding"] for d in r["data"]]


class OpenAIEmbedder(Embedder):
    def _client(self):
        if self.cfg.provider == "azure":
            from openai import AzureOpenAI
            return AzureOpenAI(api_key=self.cfg.api_key,
                               azure_endpoint=self.cfg.api_base,
                               api_version=self.cfg.api_version or "2024-06-01")
        from openai import OpenAI
        kw = {"api_key": self.cfg.api_key or "EMPTY"}
        if self.cfg.api_base:
            kw["base_url"] = self.cfg.api_base
        return OpenAI(**kw)

    def embed(self, texts):
        client = self._client()
        model = self.cfg.deployment or self.cfg.model
        r = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in r.data]


def make_embedder(cfg) -> Embedder:
    if cfg.provider == "litellm":
        return LiteLLMEmbedder(cfg)
    if cfg.provider in ("openai", "azure", "poe"):
        return OpenAIEmbedder(cfg)
    raise ValueError(f"嵌入不支持的 provider: {cfg.provider}")


class DenseIndex:
    NAME = "dense"

    def __init__(self, embedder: Embedder, batch: int = 64):
        self.embedder = embedder
        self.batch = batch
        self.pids: List[int] = []
        self.mat: List[List[float]] = []   # 已 L2 归一
        self._np = _try_numpy()

    @staticmethod
    def _norm(v):
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def build(self, corpus: List[Tuple[int, str]], verbose=False):
        self.pids, self.mat = [], []
        for i in range(0, len(corpus), self.batch):
            chunk = corpus[i:i + self.batch]
            vecs = self.embedder.embed([t for _, t in chunk])
            for (pid, _), v in zip(chunk, vecs):
                self.pids.append(pid)
                self.mat.append(self._norm(v))
            if verbose:
                print(f"[semantic] dense 嵌入 {min(i+self.batch,len(corpus))}/{len(corpus)}",
                      file=sys.stderr)
        if self._np is not None and self.mat:
            self.mat = self._np.array(self.mat, dtype="float32")

    def _qvec(self, text: str):
        v = self._norm(self.embedder.embed([text])[0])
        return self._np.array(v, dtype="float32") if self._np is not None else v

    def query(self, text: str, topk: int = 50):
        if len(self.pids) == 0:
            return []
        q = self._qvec(text)
        if self._np is not None:
            sims = self.mat @ q
            idx = sims.argsort()[::-1][:topk]
            return [(self.pids[i], float(max(0.0, min(1.0, sims[i])))) for i in idx]
        sims = [(self.pids[j], sum(a * b for a, b in zip(row, q)))
                for j, row in enumerate(self.mat)]
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return [(pid, max(0.0, min(1.0, s))) for pid, s in sims[:topk]]

    def scores(self, text: str, pids: List[int]) -> Dict[int, float]:
        allsc = dict(self.query(text, topk=len(self.pids)))
        return {p: allsc.get(p, 0.0) for p in pids}

    def to_blob(self):
        mat = self.mat.tolist() if self._np is not None and hasattr(self.mat, "tolist") \
            else self.mat
        return {"pids": self.pids, "mat": mat}

    def from_blob(self, blob):
        self.pids = blob["pids"]
        self.mat = blob["mat"]
        if self._np is not None and self.mat:
            self.mat = self._np.array(self.mat, dtype="float32")


def _try_numpy():
    try:
        import numpy
        return numpy
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 构建 / 缓存
# ---------------------------------------------------------------------------

def _sig(kind: str, model: str, n: int) -> str:
    h = hashlib.md5(f"{kind}|{model}|{n}".encode()).hexdigest()[:12]
    return f"{kind}_{n}_{h}.pkl"


def build_index(corpus_rows, semantic_cfg, llm_for_embed=None, verbose=False):
    """corpus_rows: [(pid, text)]。返回构建好的 SemanticIndex,带磁盘缓存。

    semantic_cfg.provider: tfidf(默认离线) | litellm | openai | azure
    """
    provider = semantic_cfg.provider
    n = len(corpus_rows)
    if provider == "tfidf":
        cache = _cache_dir() / _sig("tfidf", "ng23", n)
        idx = TfidfIndex()
        if cache.is_file():
            idx.from_blob(pickle.loads(cache.read_bytes()))
            if idx.n_docs == n:
                return idx
        idx.build(corpus_rows, verbose=verbose)
        cache.write_bytes(pickle.dumps(idx.to_blob()))
        return idx
    # 神经嵌入
    model = llm_for_embed.model if llm_for_embed else semantic_cfg.model
    cache = _cache_dir() / _sig("dense", model or provider, n)
    idx = DenseIndex(make_embedder(llm_for_embed))
    if cache.is_file():
        idx.from_blob(pickle.loads(cache.read_bytes()))
        if len(idx.pids) == n:
            return idx
    idx.build(corpus_rows, verbose=verbose)
    cache.write_bytes(pickle.dumps(idx.to_blob()))
    return idx
