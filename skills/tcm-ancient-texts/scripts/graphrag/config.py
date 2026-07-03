"""配置解析:环境变量 + 可选 JSON 配置文件 + 每角色模型覆盖。

优先级(高→低):CLI 显式参数 > 配置文件 > 环境变量 > 内置默认。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# LLM 判断角色(与 pipeline 实际创建的 client 一一对应)。
# 加权重排是确定性公式;reranker 是其后可选的 LLM 交叉编码器精排(不改变可复现的
# 加权分,仅在其上做一次相关性精排,provider=rule 时自动跳过)。
ROLES = ("analyzer", "extractor", "normalizer", "reranker", "judge", "verifier")


@dataclass
class LLMConfig:
    provider: str = "rule"          # rule | litellm | azure | poe | openai
    model: str = ""                 # 默认模型(各 provider 语义不同)
    api_key: str = ""
    api_base: str = ""              # openai/poe base_url;azure endpoint
    api_version: str = ""           # azure 专用
    deployment: str = ""            # azure 部署名(覆盖 model)
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    extra: dict = field(default_factory=dict)   # 透传给底层 SDK 的额外参数
    # 每角色模型覆盖: {"judge": "gpt-4o", "extractor": "gpt-4o-mini"}
    role_models: dict = field(default_factory=dict)

    def for_role(self, role: str) -> "LLMConfig":
        """返回某角色的有效配置(可能覆盖 model)。"""
        model = self.role_models.get(role) or self.model
        return LLMConfig(
            provider=self.provider, model=model, api_key=self.api_key,
            api_base=self.api_base, api_version=self.api_version,
            deployment=self.deployment, temperature=self.temperature,
            max_tokens=self.max_tokens, timeout=self.timeout, extra=dict(self.extra))


@dataclass
class SemanticConfig:
    provider: str = "tfidf"         # off | tfidf(离线默认) | litellm | openai | azure
    model: str = ""                 # 神经嵌入模型(如 text-embedding-3-small / bge-m3)
    topk: int = 40                  # 语义路由召回条数
    # 神经嵌入的凭据默认复用主 LLM 配置;如需独立指定可在配置文件 semantic.llm 覆盖
    llm: Optional[LLMConfig] = None


@dataclass
class EngineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    topk_recall: int = 60           # 召回候选上限
    topk_cards: int = 10            # 最终证据卡片数
    domain: str = "osteoporosis"    # 当前病种本体
    weights: dict = field(default_factory=dict)   # 覆盖 rerank 权重
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    llm_rerank: bool = True         # 是否启用 LLM 交叉编码器精排(provider=rule 时自动跳过)
    llm_rerank_topn: int = 20       # 送入 LLM 精排的候选数


_DEFAULT_MODELS = {
    "litellm": "gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "poe": "GPT-4o-Mini",
    "azure": "",                    # 由 deployment 决定
    "rule": "rule-heuristic",
}


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> EngineConfig:
    """加载配置。path 指向 JSON;overrides 为 CLI 传入的键值(最高优先级)。"""
    cfg = EngineConfig()
    llm = cfg.llm

    # 1) 通用环境变量(provider 专属凭据留到 provider 最终确定后再读,见第 4 步)
    llm.provider = _env("TCM_LLM_PROVIDER", default="rule").lower()
    llm.model = _env("TCM_LLM_MODEL")
    llm.temperature = float(_env("TCM_LLM_TEMPERATURE", default="0") or 0)
    llm.max_tokens = int(_env("TCM_LLM_MAX_TOKENS", default="1024") or 1024)

    # 2) 配置文件(覆盖环境变量)
    path = path or _env("TCM_GRAPHRAG_CONFIG")
    if path and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _apply(cfg, data)

    # 3) CLI overrides(最高)
    if overrides:
        _apply(cfg, overrides)
    llm.provider = (llm.provider or "rule").lower()   # 文件/CLI 传入也规范化

    # 4) provider 专属凭据/端点(此时 provider 已最终确定;仅填补仍为空的字段,
    #    使 --provider / 配置文件指定 provider 时也能从环境变量取到凭据)
    _fill_provider_env(llm)
    if not llm.model:
        llm.model = _DEFAULT_MODELS.get(llm.provider, "")
    return cfg


def _fill_provider_env(llm: LLMConfig):
    """按最终 provider 从环境变量补全空缺的凭据/端点。已由文件/CLI 设置的不覆盖。"""
    def fill(attr, *names, default=""):
        if not getattr(llm, attr):
            v = _env(*names, default=default)
            if v:
                setattr(llm, attr, v)

    if llm.provider == "azure":
        fill("api_key", "AZURE_OPENAI_API_KEY", "TCM_LLM_API_KEY")
        fill("api_base", "AZURE_OPENAI_ENDPOINT", "TCM_LLM_API_BASE")
        fill("api_version", "AZURE_OPENAI_API_VERSION", default="2024-06-01")
        fill("deployment", "AZURE_OPENAI_DEPLOYMENT", "TCM_LLM_MODEL")
    elif llm.provider == "poe":
        fill("api_key", "POE_API_KEY", "TCM_LLM_API_KEY")
        fill("api_base", "POE_API_BASE", default="https://api.poe.com/v1")
    elif llm.provider == "openai":
        fill("api_key", "OPENAI_API_KEY", "TCM_LLM_API_KEY")
        fill("api_base", "OPENAI_BASE_URL", "TCM_LLM_API_BASE")
    elif llm.provider == "litellm":
        fill("api_key", "TCM_LLM_API_KEY", "OPENAI_API_KEY")
        fill("api_base", "TCM_LLM_API_BASE")


def _apply(cfg: EngineConfig, data: dict):
    llm_data = data.get("llm", {})
    for k, v in llm_data.items():
        if hasattr(cfg.llm, k) and v is not None:
            setattr(cfg.llm, k, v)
    for k in ("topk_recall", "topk_cards", "domain", "weights",
              "llm_rerank", "llm_rerank_topn"):
        if data.get(k) is not None:
            setattr(cfg, k, data[k])
    sem = data.get("semantic")
    if sem is not None:
        for k in ("provider", "model", "topk"):
            if sem.get(k) is not None:
                setattr(cfg.semantic, k, sem[k])
        if sem.get("llm"):   # 独立的嵌入凭据
            elc = LLMConfig()
            for k, v in sem["llm"].items():
                if hasattr(elc, k) and v is not None:
                    setattr(elc, k, v)
            cfg.semantic.llm = elc
