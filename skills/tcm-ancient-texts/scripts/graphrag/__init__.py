"""tcm-graphrag — 中医古籍表型证据 GraphRAG 引擎。

在纯检索地基 (tcm.py) 之上叠加:古今术语本体、四路召回、加权重排、
五重模型角色(Extractor/Normalizer/Reranker/EvidenceJudge/Verifier)与证据卡片。

LLM 后端统一抽象,支持 litellm / azure / poe / openai,以及无需 API key 的
规则离线 provider(rule)。核心引擎仅依赖 Python 标准库;各 LLM SDK 懒加载。
"""

__version__ = "0.1.0"
