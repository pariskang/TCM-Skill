"""配置解析:环境变量 + 可选 JSON 配置文件 + 每角色模型覆盖。

优先级(高→低):CLI 显式参数 > 配置文件 > 环境变量 > 内置默认。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 五个模型角色
ROLES = ("extractor", "normalizer", "reranker", "judge", "verifier")


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
class EngineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    topk_recall: int = 60           # 召回候选上限
    topk_cards: int = 10            # 最终证据卡片数
    domain: str = "osteoporosis"    # 当前病种本体
    weights: dict = field(default_factory=dict)   # 覆盖 rerank 权重


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

    # 1) 环境变量
    llm.provider = _env("TCM_LLM_PROVIDER", default="rule").lower()
    llm.model = _env("TCM_LLM_MODEL")
    llm.temperature = float(_env("TCM_LLM_TEMPERATURE", default="0") or 0)
    llm.max_tokens = int(_env("TCM_LLM_MAX_TOKENS", default="1024") or 1024)

    # provider 专属环境变量
    if llm.provider == "azure":
        llm.api_key = _env("AZURE_OPENAI_API_KEY", "TCM_LLM_API_KEY")
        llm.api_base = _env("AZURE_OPENAI_ENDPOINT", "TCM_LLM_API_BASE")
        llm.api_version = _env("AZURE_OPENAI_API_VERSION", default="2024-06-01")
        llm.deployment = _env("AZURE_OPENAI_DEPLOYMENT", "TCM_LLM_MODEL")
    elif llm.provider == "poe":
        llm.api_key = _env("POE_API_KEY", "TCM_LLM_API_KEY")
        llm.api_base = _env("POE_API_BASE", default="https://api.poe.com/v1")
    elif llm.provider == "openai":
        llm.api_key = _env("OPENAI_API_KEY", "TCM_LLM_API_KEY")
        llm.api_base = _env("OPENAI_BASE_URL", "TCM_LLM_API_BASE")
    elif llm.provider == "litellm":
        llm.api_key = _env("TCM_LLM_API_KEY", "OPENAI_API_KEY")
        llm.api_base = _env("TCM_LLM_API_BASE")

    # 2) 配置文件(覆盖环境变量)
    path = path or _env("TCM_GRAPHRAG_CONFIG")
    if path and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _apply(cfg, data)

    # 3) CLI overrides(最高)
    if overrides:
        _apply(cfg, overrides)

    # 4) provider 默认兜底(在 provider 最终确定后再补默认端点/模型)
    if llm.provider == "poe" and not llm.api_base:
        llm.api_base = "https://api.poe.com/v1"
    if llm.provider == "azure" and not llm.api_version:
        llm.api_version = "2024-06-01"
    if not llm.model:
        llm.model = _DEFAULT_MODELS.get(llm.provider, "")
    return cfg


def _apply(cfg: EngineConfig, data: dict):
    llm_data = data.get("llm", {})
    for k, v in llm_data.items():
        if hasattr(cfg.llm, k) and v is not None:
            setattr(cfg.llm, k, v)
    for k in ("topk_recall", "topk_cards", "domain", "weights"):
        if data.get(k) is not None:
            setattr(cfg, k, data[k])
