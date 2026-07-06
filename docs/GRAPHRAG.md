# 中医古籍表型证据 GraphRAG 系统

> Ancient TCM Phenotype-Evidence GraphRAG System
> 从"古籍关键词搜索"升级为"古籍—术语—现代表型—证候—方药—证据链的可解释检索判断系统"。

输入一个现代疾病 / 表型 / 证型 / 症状组合 / 研究假说,系统返回**最相关古籍条文 +
古今表型映射 + 证候-病机-治法-方药证据链 + 证据等级 + 模型判断理由 + 事实核验 +
专家可复核出处**——每条结果是一张 **Evidence Card(证据卡片)**,而非一段自由文本。

它回答的不是"哪条文献提到了骨痿",而是:
> **哪一条古籍原文,在什么上下文中,通过什么术语、表型、证候、治法和方药证据,
> 可以在多大程度上支持某个现代疾病或研究假说。**

> **方法学依据**:检索融合(HippoRAG 2 的 PPR、RRF)、引用忠实度核验
> (ALCE / FActScore / RAGAS)等设计的文献支撑见 [`docs/RESEARCH.md`](RESEARCH.md)
> ——每项决策对齐到 2024–2026 经三票对抗核验存活的顶会结论,并诚实标注
> 未验证方向(古汉语 NLP / ICD-11 TM / GRADE)。

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
查询理解 → 四路召回 → 加权重排 → LLM 精排 → 模型裁判 → 证据卡片
  │           │           │          │          │
  │           │           │          │          ├ Extractor   实体抽取 + evidence_span
  │           │           │          │          ├ Normalizer  古今术语映射 + 匹配类型
  │           │           │          │          ├ EvidenceJudge 证据等级 A-E + 理由
  │           │           │          │          └ Verifier    幻觉核验(断言↔原文)
  │           │           │          └ Reranker  交叉编码器式 LLM 相关性精排(可选;
  │           │           │             与确定性分 5:5 融合;provider=rule 时跳过)
  │           │           └ S_final 加权(lexical/semantic/RRF/ontology/phenotype/
  │           │              context/evidence/dynasty − exclusion),确定性、可复现
  │           ├ lexical  FTS5 trigram 精确短语(内建 BM25 排序)+ LIKE(2 字词按书轮取)
  │           ├ synonym  本体同义/异体扩展
  │           ├ graph    图谱路径:PPR(Personalized PageRank)对证候种子做带重启
  │           │          随机游走,优先检索"离种子近、多路径汇聚"的表型/治法/方药
  │           └ semantic 向量语义:tfidf(离线默认)或神经嵌入(litellm/openai/azure)
  └ QueryAnalyzer:简繁归一 + 拆解为 现代疾病/现代表型/中医证候/古籍检索词/证候种子
```

每条召回列表内的**名次**被完整记录到候选,重排层用 **RRF(Reciprocal Rank
Fusion,k=60)** 融合为"多路共识"子分——只用名次不用分值,天然免疫 BM25 rank/
余弦/LIKE 密度代理三种分值尺度不可比的问题;被多条独立检索路径都排在前面的
条文得分更高。图谱路由的 PPR 取代了原先"BFS 逐跳一视同仁"的扩展方式,
多跳目标按图结构相关度几何衰减(HippoRAG 验证过的图谱召回打分方式)。

**六个 LLM 角色** = QueryAnalyzer + Extractor + Normalizer + **Reranker** +
EvidenceJudge + Verifier(均可用 `role_models` 分别指定模型;provider=rule 时全部
走确定性规则)。加权重排是确定性公式(可复现);其后的 Reranker 是**可选**的 LLM
交叉编码器精排,与加权分 5:5 融合。四路召回全部已实现。

## 语义召回(第四路)

两种后端,统一接口(`embeddings.py`):

| 后端 | 依赖 | 说明 |
|---|---|---|
| `tfidf`(默认) | 无 | 字符 n-gram(2+3)TF-IDF + 倒排索引 + 余弦。离线、零依赖、确定性、可缓存;能召回"用词相近但非精确子串"的条文(如查"骨弱不能行走"召回"虛弱不能行"条文) |
| `litellm`/`openai`/`azure` | 对应 SDK | 神经嵌入(如 text-embedding-3-small / bge-m3),语义级;大规模建议装 numpy 加速余弦 |

- 索引在首次 `ask` 时构建并**磁盘缓存**(`data/semantic/`,按段落数指纹),重建即失效;
- 语义分同时用于**召回**(补齐精确匹配漏掉的条文)与**重排**(S_final 的 semantic 子分);
- 简体查询在语义检索前先转繁体(语料为繁体),否则召不回;
- `--semantic off` 关闭,`--semantic openai --embed-model text-embedding-3-small` 用神经嵌入。

## LLM 交叉编码器精排(Reranker)

加权重排取 top-N(默认 20)后,把 `查询 × 每条原文` 批量交给 LLM 逐条打相关性分
(0~1),识别"貌似相关实则语境不符"的条文(危候/外伤/他病),与确定性加权分 5:5
融合后重排。`provider=rule` 或 `--no-llm-rerank` 时跳过,保持纯确定性排序。

## 简繁完美支持

三重保障:opencc(若 `pip install opencc-python-reimplemented`,完整权威)→ 内置
高频字表(离线兜底,~200 字覆盖中医术语)→ LLM 分析器(有 LLM 后端时语义级转换,
处理一简多繁与口语)。查询、语义检索、术语匹配全链路统一用繁体。

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

## 排除机制:句级共现判定(证据系统的关键)

朴素检索会把"大骨枯槁,真臟脈見,期六月死"(《素问·玉机真臟论》的**危重预后**)
误当作"骨枯=骨质疏松"证据。本系统的排除标签会识别 `真臟脈見/期六月死/大肉陷下`,
在重排中扣分、并由裁判判为 **E 级·已排除**。

但排除不能是"全段关键词一票否决"——古籍长段常同时含定义性论述与其他语境:
《素问·痿论》既有核心条文"腎氣熱…骨枯而髓減,發為骨痿",又有肺痿论述
"肺熱葉焦"(后者是排除词)。因此排除采用**句级共现判定**
(`Ontology.exclusions_scoped`):

- **硬排除**:段内每一个含核心阳性词(L1 病名/L3 表型)的句子都同时含排除词
  ——核心词本身处于被排除语境 → 重排全额扣分,裁判判 E。
  (《玉机真臟论》:"大骨枯槁…期六月死"同句 → E ✓)
- **软排除**:存在至少一个"干净"的核心词句,排除词只在他句——上下文噪声
  → 扣分打 0.35 折,不判 E;若裁判本判 A 则降为 B 并注明"待复核"。
  (《痿论》:定义性条文不再被误杀,正常评级 ✓)

两个方向都进了金标准回归用例(A01/X01/X02),任何回归立即暴露。

## 方证极性:处方 vs 禁忌(否定/禁忌检测)

古籍(尤以《伤寒论》)对**同一方剂在不同证候下立场相反**,朴素检索把它们当同等
相关是科学错误:

| 原文 | 立场 |
|---|---|
| 太陽病…**桂枝湯主之**(段159) | 处方(该用)|
| 若酒客病,**不可與桂枝湯**(段166) | 禁忌(禁用)|
| 壞病,**桂枝不中與之**也(段165) | 禁忌 |
| 可與大承氣湯…**不可與之**(段584) | 辨证使用(条件)|

`graphrag/negation.py` 是临床 NLP 否定检测经典算法(**NegEx**, Chapman et al.,
J Biomed Inform 2001;**ConText**, Harkema et al., J Biomed Inform 2009)对文言
中医文本的适配:极性是**每个方剂目标**的属性,由**否定线索词 + 分句作用域**决定
——**一段可同时肯定 A 方、否定 B 方**(段584:大承气条件、小承气处方;段463:柴胡
条件、半夏泻心汤处方)。聚合规则:全否定→`negate`、全肯定→`affirm`、既肯定又
否定→`conditional`(辨证使用)、仅提及→`neutral`。

落地三处:
- **证据卡片**:新增「方证极性」徽章(✅处方 / ⛔禁忌 / ⚖️辨证使用),并列出命中线索;
- **重排**:`polarity` 惩罚子分(禁忌全罚、辨证半罚)——被禁忌的方剂作为"该方治此病"
  的证据更弱,但不剔除(禁忌本身是有价值的反证);
- **裁判**:方剂被明确禁忌时封顶 C 级(反证/背景),辨证使用封顶 B 级。

全部 5 个《伤寒论》复杂案例(含混合极性 段584/463)进金标准回归(P01–P05);
且该能力**跨域通用**——已验证《景岳全书》"其有生平不宜熟地者"正确判为熟地禁忌。

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
    "role_models": {"judge": "gpt-4o", "verifier": "gpt-4o", "reranker": "gpt-4o-mini"}
  },
  "semantic": {"provider": "openai", "model": "text-embedding-3-small", "topk": 40},
  "llm_rerank": true,
  "llm_rerank_topn": 20,
  "topk_cards": 10
}
```

语义嵌入默认复用主 LLM 凭据;如需用不同 key/端点做嵌入,在 `semantic.llm` 单独指定。

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
python3 tcm_graphrag.py eval                   # 金标准回归评测(CI 可用,exit 1 即回归)
```

## 加权重排公式

```
S_final = 0.15·lexical + 0.15·semantic + 0.10·rrf + 0.15·ontology
        + 0.15·phenotype + 0.10·context + 0.10·evidence + 0.05·dynasty
        − 0.05·exclusion − 0.05·polarity
```

- `rrf` = 各召回列表名次的 Reciprocal Rank Fusion(k=60),按候选集内最大值
  归一;度量"多路召回共识"。
- `exclusion` 按句级共现判定:硬排除全额扣分,软排除打 0.35 折(见排除机制)。
- `polarity` 惩罚:方剂被禁忌(negate)全额扣分、辨证使用(conditional)半罚
  (见方证极性)。
- 未启用 semantic(无向量)时,其权重按 6:4 自动分摊给 lexical/ontology,
  保证分值可比。各子分含义见 `graphrag/rerank.py`。权重可在配置文件
  `weights` 覆盖(含新增的 `rrf` 键)。

## 金标准回归评测(eval)

`eval/gold.jsonl` 收录**经 tcm.py 实检验证**的金标准用例(检索/端到端/排除
三类),`tcm_graphrag.py eval` 用 provider=rule + tfidf 的全确定性路径回归:

| 用例类型 | 验证什么 | 指标 |
|---|---|---|
| search | 检索地基:期望书目/段落在 top-k | Recall@k、MRR |
| ask | 端到端:期望条文入卡、最低等级约束、**全部 evidence_span 过引用忠实度核验** | Recall@k、MRR、span 违例数 |
| exclusion | 危候等禁忌片段只允许 E 级 | 排除正确率 |
| polarity | 方剂处方/禁忌/辨证使用判定(《伤寒论》复杂案例) | 极性正确率 |

汇总额外输出 **RAGAS 式免参考忠实度** `F=|V|/|S|`(Es et al., EACL 2024):
S=卡片抽取的全部要素,V=其中确有原文支撑者。任一用例失败即 exit 1,
可直接作 CI 回归门。当前 **17/17 通过,平均 Recall@10 = 1.000,F = 1.000**
(含 5 个《伤寒论》方证极性用例 P01–P05)。

> 上述所有设计的文献依据(HippoRAG 2 / LightRAG / RRF / ALCE / FActScore /
> CoVe / RAGAS 等,均经三票对抗核验)见 [`docs/RESEARCH.md`](RESEARCH.md);
> 完整核验记录见 `docs/research-notes/graphrag-sota-2026.md`。

## 引用忠实度:双层确定性防线

LLM 抽取的 `evidence_span` 可能是改写而非原文(attributable generation 的
经典失败模式)。防线有两层,全部确定性、不依赖 LLM 自查:

1. **源头修复**(`agents.extract`):span 非原文子串(忽略空白)→ 立即替换为
   规则法选出的真实句子;
2. **纵深防御**(`agents.verify`):核验入口先做子串检查,不通过直接
   `grounded=False`,LLM 无权放行。

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

## 已完成的核心能力

- ✅ **语义召回**:tfidf 离线倒排 + 神经嵌入(litellm/openai/azure),磁盘缓存,
  同时用于召回与重排子分。
- ✅ **LLM 交叉编码器精排**:top-N 批量相关性精排,与确定性分 5:5 融合。
- ✅ **简繁完美支持**:opencc 可选 + 内置表 + LLM 分析器三重保障。
- ✅ **RRF 多路融合**:四路召回名次的 Reciprocal Rank Fusion 子分(k=60)。
- ✅ **PPR 图谱召回**:Personalized PageRank 对图谱扩展目标打分排序。
- ✅ **句级共现排除**:硬/软排除区分,定义性条文不再被"全段一票否决"误杀。
- ✅ **方证极性(否定/禁忌检测)**:NegEx/ConText 式,分句作用域、按方剂,
  处方/禁忌/辨证使用三分;卡片徽章 + 重排惩罚 + 裁判降级。
- ✅ **引用忠实度双层核验**:evidence_span 源头修复 + 核验入口确定性拦截。
- ✅ **金标准回归评测**:`eval` 命令,17 用例全确定性回归(含《伤寒论》
  方证极性),CI 可用。

## 扩展路线

1. **MCP Server**:把 `ask` 封装为 MCP tool,接入 Claude Desktop / 任意 MCP 客户端。
2. **多病种**:银屑病、认知障碍——各加一组 `ontology/*.<domain>.json`。
3. **结构化抽取库**:方剂表(方名-组成-剂量-主治-出处)入 SQLite 新表。
4. **全量语料向量化加速**:百万级段落建议神经嵌入 + numpy/faiss;tfidf 倒排在
   全量下内存占用需评估(当前每段截 top-160 特征已做界定)。
5. **金标准扩容**:随病种/语料扩展持续把实检验证过的用例加入 gold.jsonl。

## 目录

```
scripts/
├── tcm.py                    # 检索地基(纯标准库,不动)
├── tcm_graphrag.py           # GraphRAG CLI 入口
├── requirements.txt          # 可选 LLM SDK
├── graphrag/
│   ├── config.py             # 配置解析(env/文件/CLI,每角色模型,语义/精排开关)
│   ├── llm.py                # LLM 客户端抽象(litellm/azure/poe/openai)
│   ├── zh.py                 # 简繁转换(opencc 可选 + 内置表 + LLM)
│   ├── corpus.py             # 只读语料访问(复用 tcm.py 索引)
│   ├── ontology.py           # 本体加载/同义扩展/句级排除/图谱 BFS+PPR
│   ├── negation.py           # 方证极性:NegEx/ConText 式否定/禁忌检测(处方/禁忌/辨证)
│   ├── embeddings.py         # 语义后端:tfidf 倒排(离线)+ 神经嵌入 + 磁盘缓存
│   ├── recall.py             # 四路召回(lexical/synonym/graph/semantic)+ 名次记录
│   ├── rerank.py             # 加权重排(确定性,含 RRF 子分 + 极性惩罚)
│   ├── agents.py             # 六个 LLM 角色 + 查询解析(LLM/规则双路)+ span 核验
│   ├── evidence.py           # 证据对象 + 证据卡片渲染(含方证极性徽章)
│   ├── evaluate.py           # 金标准回归评测(Recall@k/MRR/排除/极性/忠实度)
│   └── pipeline.py           # 编排
├── eval/
│   └── gold.jsonl            # 金标准用例(实检验证后录入)
└── ontology/
    ├── terms.osteoporosis.json      # L1-L7 术语
    ├── bridge.osteoporosis.json     # 古今表型桥接矩阵 + M1-M5
    └── relations.osteoporosis.json  # 知识图谱边
```
