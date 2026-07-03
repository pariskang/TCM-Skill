# 中医古籍表型证据 GraphRAG 系统

> Ancient TCM Phenotype-Evidence GraphRAG System
> 从"古籍关键词搜索"升级为"古籍—术语—现代表型—证候—方药—证据链的可解释检索判断系统"。

输入一个现代疾病 / 表型 / 证型 / 症状组合 / 研究假说,系统返回**最相关古籍条文 +
古今表型映射 + 证候-病机-治法-方药证据链 + 证据等级 + 模型判断理由 + 事实核验 +
专家可复核出处**——每条结果是一张 **Evidence Card(证据卡片)**,而非一段自由文本。

它回答的不是"哪条文献提到了骨痿",而是:
> **哪一条古籍原文,在什么上下文中,通过什么术语、表型、证候、治法和方药证据,
> 可以在多大程度上支持某个现代疾病或研究假说。**

## 与基础检索(tcm.py)的关系

| | tcm.py | graphrag |
|---|---|---|
| 定位 | 确定性全文检索地基 | 证据推理层 |
| 依赖 | 纯标准库 | 标准库 + 可选 LLM SDK |
| 输入 | 繁体检索词 | 现代自然语言查询 |
| 输出 | 带出处的条文片段 | 证据卡片(等级/映射/链/核验) |

GraphRAG **复用** tcm.py 的 SQLite FTS5 索引(只读),不重复建库。先 `tcm.py
fetch && build`,再用 `tcm_graphrag.py ask`。

## 五层架构

```
查询理解 → 多路召回 → 加权重排(确定性)→ 模型裁判 → 证据卡片
  │           │           │                 │
  │           │           │                 ├ Extractor   实体抽取 + evidence_span
  │           │           │                 ├ Normalizer  古今术语映射 + 匹配类型
  │           │           │                 ├ EvidenceJudge 证据等级 A-E + 理由
  │           │           │                 └ Verifier    幻觉核验(断言↔原文)
  │           │           └ S_final 加权(lexical/semantic/ontology/phenotype/
  │           │              context/evidence/dynasty − exclusion);刻意用确定性
  │           │              公式而非 LLM,以保证可复现(LLM 精排见路线图)
  │           ├ lexical  FTS5 trigram 精确短语        ┐
  │           ├ synonym  本体同义/异体扩展            ├ 已实现
  │           ├ graph    图谱路径:证候→表型/治法/方药→回检语料 ┘
  │           └ semantic (预留)向量语义,未配置则跳过不伪造,见路线图
  └ QueryAnalyzer:拆解为 现代疾病 / 现代表型 / 中医证候 / 古籍检索词 / 证候种子
```

**五个 LLM 判断角色** = QueryAnalyzer + Extractor + Normalizer + EvidenceJudge +
Verifier(均可用 `role_models` 分别指定模型)。重排(Reranker)刻意采用确定性
加权公式而非 LLM,是有意的设计选择——保证同一查询结果可复现;"LLM 交叉编码器精排"
作为可选增强列在路线图。当前召回三路已实现(lexical/synonym/graph),semantic 预留。

## 本体:古今双向映射

- **L1–L7 古籍本体**(`ontology/terms.<domain>.json`):古病名 / 证候病机 /
  症状表型 / 人群锚点 / 治法 / 方药 / 结局。每个术语带同义词与**排除标签**。
- **M1–M5 现代层**(`ontology/bridge.<domain>.json` 的 modern_layers):
  现代疾病 / 现代表型 / 客观指标 / 临床背景 / 机制实验。
- **古今表型桥接矩阵**(bridges):每行 = 现代维度 → 现代表型 → 中医证候 →
  古籍候选表述 → **匹配等级**(exact/broad/narrow/related/negativeMatch)。
- **知识图谱边**(`ontology/relations.<domain>.json`):`证候-manifests→表型`、
  `病名-treated_by→治法`、`治法-uses_herb→方药`,用于图谱路径召回。

当前内置病种:`osteoporosis`(骨质疏松 MVP)。新增病种只需照格式加三个 JSON 文件,
`domains` 命令自动发现。

## 排除机制:证据系统的关键

朴素检索会把"大骨枯槁,真臟脈見,期六月死"(《素问·玉机真臟论》的**危重预后**)
误当作"骨枯=骨质疏松"证据。本系统的排除标签会识别 `真臟脈見/期六月死/大肉陷下`,
在重排中扣分、并由裁判判为 **E 级·已排除**。这种"假阳性鉴别"正是它区别于关键词
搜索之处。(已在样本语料上验证。)

## LLM 后端:统一抽象,四家可切换 + 离线兜底

| provider | 依赖 | 用途 |
|---|---|---|
| `rule` | 无(默认) | 离线规则引擎,本体驱动的确定性抽取/判断,无需 API key |
| `litellm` | `pip install litellm` | 统一路由百余后端(gpt/claude/gemini/…) |
| `azure` | `pip install openai` | Azure OpenAI(AzureOpenAI 客户端) |
| `poe` | `pip install openai` | Poe OpenAI 兼容端点 `https://api.poe.com/v1` |
| `openai` | `pip install openai` | 任意 OpenAI 兼容端点(含自建/vLLM) |

**设计要点**:`rule` 让系统零依赖、零成本、离线可跑、结果确定可复现(用于 CI 回归
与无网环境);配置真实 LLM 后端即无缝提升抽取/归一/裁判质量。SDK **懒加载**,
未安装不影响 import 与 rule 模式。

### 配置(优先级:CLI > 配置文件 > 环境变量 > 默认)

环境变量:

```bash
# 通用
export TCM_LLM_PROVIDER=poe            # rule|litellm|azure|poe|openai
export TCM_LLM_MODEL=Claude-Sonnet-4.5
export TCM_LLM_TEMPERATURE=0

# Azure
export AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
export AZURE_OPENAI_API_VERSION=2024-06-01  AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Poe
export POE_API_KEY=...

# OpenAI / 兼容端点
export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.openai.com/v1
```

配置文件(`--config-file cfg.json`),支持**每角色不同模型**——便宜模型做抽取、
强模型做裁判:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "role_models": {"judge": "gpt-4o", "verifier": "gpt-4o"}
  },
  "topk_cards": 10
}
```

## 用法

```bash
cd skills/tcm-ancient-texts/scripts

# 离线(默认,无需 key)
python3 tcm_graphrag.py ask "绝经后骨质疏松 肾虚血瘀 骨痛 活动受限"

# 接入 Poe
TCM_LLM_PROVIDER=poe POE_API_KEY=xxx \
  python3 tcm_graphrag.py ask "骨痿 腰膝酸软 老年" --model Claude-Sonnet-4.5

# 接入 Azure
python3 tcm_graphrag.py ask "肾虚骨痿" --provider azure --deployment gpt-4o

# JSON 输出(供下游/前端消费)
python3 tcm_graphrag.py ask "骨枯 腰痛" --format json

# 辅助
python3 tcm_graphrag.py providers --check      # 连通性自检
python3 tcm_graphrag.py domains                # 可用病种
python3 tcm_graphrag.py config                 # 生效配置
```

## 加权重排公式

```
S_final = 0.20·lexical + 0.20·semantic + 0.15·ontology + 0.15·phenotype
        + 0.10·context + 0.10·evidence + 0.05·dynasty − 0.05·exclusion
```

未启用 semantic(无向量)时,其权重按 6:4 自动分摊给 lexical/ontology,保证分值可比。
各子分含义见 `graphrag/rerank.py`。权重可在配置文件 `weights` 覆盖。

## 证据等级

| 等级 | 定义 | 用途 |
|---|---|---|
| A | 古病名/表型/病机/治法高度一致,链条完整 | 核心证据 |
| B | 表型病机一致,病名或链条不全 | 支持证据 |
| C | 仅单一症状或宽泛相关 | 背景证据 |
| D | 牵强、歧义大、易误配 | 不建议纳入 |
| E | 命中排除项,明确排除 | 剔除 |

## 与专家闭环

模型/规则给出的等级是**初判**;每张卡片带 `expert_status`(未审/已审/一致/争议/
剔除)与完整出处、evidence_span、判断理由、排除检查,供专家 15% 抽样复核
(κ 一致性)。模型负责建议,专家负责确认。

## 扩展路线

1. **语义召回**:`tcm.py build --jsonl` → 段落 embedding(bge-m3)→ 与 FTS5 做
   RRF 混合;在 `recall.py` 接入 `semantic_scores`,`rerank.py` 打开 `semantic_enabled`。
2. **MCP Server**:把 `ask` 封装为 MCP tool,接入 Claude Desktop / 任意 MCP 客户端。
3. **多病种**:银屑病、认知障碍——各加一组 `ontology/*.<domain>.json`。
4. **结构化抽取库**:方剂表(方名-组成-剂量-主治-出处)入 SQLite 新表。
5. **评测集**:已知出处的条文金标准问答,回归验证"始终走检索而非记忆"。

## 目录

```
scripts/
├── tcm.py                    # 检索地基(纯标准库,不动)
├── tcm_graphrag.py           # GraphRAG CLI 入口
├── requirements.txt          # 可选 LLM SDK
├── graphrag/
│   ├── config.py             # 配置解析(env/文件/CLI,每角色模型)
│   ├── llm.py                # LLM 客户端抽象(litellm/azure/poe/openai)
│   ├── corpus.py             # 只读语料访问(复用 tcm.py 索引)
│   ├── ontology.py           # 本体加载/同义扩展/排除/图谱
│   ├── recall.py             # 多路召回(lexical/synonym/graph;semantic 预留)
│   ├── rerank.py             # 加权重排
│   ├── agents.py             # 五重模型角色 + 查询解析(LLM/规则双路)
│   ├── evidence.py           # 证据对象 + 证据卡片渲染
│   └── pipeline.py           # 编排
└── ontology/
    ├── terms.osteoporosis.json      # L1-L7 术语
    ├── bridge.osteoporosis.json     # 古今表型桥接矩阵 + M1-M5
    └── relations.osteoporosis.json  # 知识图谱边
```
