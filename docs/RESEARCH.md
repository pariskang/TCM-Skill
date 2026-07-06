# 方法学依据:GraphRAG 引擎设计的文献支撑

> 本文档把 GraphRAG 引擎的每一项关键设计决策,对齐到 2024–2026 年检索增强领域
> 经**三票对抗核验**(独立三方投票、需 2/3 反驳才推翻)存活的高置信文献结论。
> 目的是让本系统的检索/融合/证据核验方法具备可复核的科学依据,而非拍脑袋。
>
> 调研范围与诚实边界见文末「未验证方向」——**方向 3–5(古汉语 NLP、ICD-11
> 传统医学模块、GRADE 分级)的候选主张未通过核验,本文不就此下结论**。

## 1. 图谱召回必须与词法/语义融合,不可独立成路

**文献**:HippoRAG 2(Gutiérrez et al., **ICML 2025**, arXiv:2502.14802);
独立评测 arXiv:2502.11371(GraphRAG 在 NQ 单跳事实 QA 上比 vanilla RAG 低约
13.4%);GraphRAG 综述 Zhang et al.(arXiv:2501.13958, 2025)。

**核心结论**:HippoRAG 2 之前的各类 KG-based GraphRAG 虽提升多跳/sense-making,
但在**基础事实记忆任务上显著低于标准向量 RAG**;HippoRAG 2 在 Personalized
PageRank 之上加入更深的段落级整合,才在事实/sense-making/关联三类任务全面反超,
关联记忆较 SOTA 稠密模型(NV-Embed-v2)+7%。原文:*"their performance on more
basic factual memory tasks drops considerably below standard RAG"*。

**本系统的落地(已实现)**:
- graph 路由用 **Personalized PageRank**(`Ontology.ppr`,damping=0.85,幂迭代)
  对证候种子做带重启随机游走,取 top-8 目标回检语料——而非独立的关键词扩展。
- graph 与 lexical/synonym/semantic 四路**在重排层融合**,并保留**段落级证据**
  (Evidence Card 绑定原文 span + 出处),契合 HippoRAG 2 的"passage-level
  integration"而非仅三元组级。
- **不让 graph 单独决定排序**:PPR 只影响 graph 路的召回顺序,最终名次由四路
  融合 + 加权重排共同决定,规避"纯图谱损害基础事实检索"的已知失效模式。

## 2. 图-向量一体化的双层检索

**文献**:LightRAG(Guo, Xia, Yu, Ao, Huang, arXiv:2410.05779, **Findings of
EMNLP 2025**)。

**核心结论**:将图结构直接并入文本索引与检索,采用**双层检索**——低层(具体
实体/关系)+ 高层(宏观主题),并将图结构与向量表征一体化。原文:*"a dual-level
retrieval system ... from both low-level and high-level knowledge discovery"*。

**本系统的落地(已实现 / 对标)**:四路召回中 lexical/synonym 对应低层实体键、
graph 对应关系与主题层、semantic 提供向量近邻;本体 L1–L7 分层天然是 LightRAG
"实体–关系–主题"的中医特化。**限定**:LightRAG 的实体抽取依赖 32B+ LLM,本系统
零依赖离线路径改用**本体/规则抽取**替代,牺牲一定召回换取确定性与可复现。

## 3. 词法通道(BM25/FTS5)在专业领域应是一等公民

**文献**:T2-RAGBench 两阶段流水线基准(arXiv:2604.01733,金融文本+表格,
23,088 查询;*medium* 置信,preprint 未评审);BEIR 零样本基准(Thakur et al.,
NeurIPS 2021)佐证"混合优于单路"。

**核心结论**:**BM25 在专业领域可单独胜过 SOTA 稠密检索**(MRR@3 0.411 vs
0.351);两阶段流水线(BM25+dense 混合召回 → 神经重排)Recall@5=0.816,大幅超
所有单阶段方法。

**本系统的落地(已实现)**:
- SQLite **FTS5 trigram 保留为词法一等公民**(内建 BM25 排序),不被嵌入替代——
  古籍方名、药名、条文编号等精确术语与金融 ticker 同理,利于词法精确匹配;
- 现有"确定性加权重排 + 可选 LLM 交叉编码器精排"与 SOTA 的"混合召回→神经重排"
  两阶段结构一致。
- **域迁移诚实声明**:上述数字来自金融/英文基准,向文言文语料是**合理类比而非
  实证**,须以自建金标准集复验(见 §6 与 eval)。

## 4. 融合方法:RRF 作基线,调优加权融合作主力

**文献**:RRF 原始工作 Cormack, Clarke & Büttcher(**SIGIR 2009**);OpenSearch
2.19(2025-02 GA)内置 RRF 混合检索;**Bruch et al., ACM TOIS 2023**——归一化
分数的凸组合在所有测试语料(域内+域外)优于 RRF;OpenSearch 自评六数据集显示
RRF 的 NDCG@10 比分数融合低 3.86%。

**核心结论**:RRF(`score=Σ 1/(k+rank)`,k=60)是**免分数归一化的稳健生产基线**,
但**调优的加权分数融合在质量上可胜过 RRF**。

**本系统的落地(已实现,且经文献校准)**:
- 把 **RRF 作为加权重排里的一个子分**(`rrf` 权重 0.10,k=60),**而非替换**整个
  确定性加权重排——正是 Bruch et al. 与 OpenSearch 自评所支持的方向:加权融合
  主力、RRF 作稳健信号与无调参回退。
- RRF 子分只用**名次**不用分值,免疫 BM25 rank / 余弦 / LIKE 密度代理三种分值
  尺度不可比的问题;度量"多路召回共识"。
- **诚实声明(两条被否决的主张)**:对抗核验**否决**了"k=60 仍是 2026 默认最佳
  实践"的引用归属(0-3)与"RRF 三方面鲁棒性优于分数融合"(1-2)两种表述。本系统
  文档因此**不主张** RRF 通用更优,仅将其定位为经典基线与共识信号。

## 5. 引用忠实度:句级 + 原子级双粒度核验

**文献**:
- ALCE(Gao, Yen, Yu & Chen, **EMNLP 2023**)——NLI 蕴含模型计算**引文召回/精确率**;
  揭示即使 GPT-4,约 **50% 生成未被所引段落完全支持**。
- FActScore(Min et al., **EMNLP 2023**)——**原子事实**分解 + 支持率;"检索+强
  LLM"自动估计器复现人工分误差 **<2%**(英文传记域)。
- Chain-of-Verification(Dhuliawala et al., **Findings of ACL 2024**)——四步
  自验证,验证问答与初稿隔离以降幻觉。
- RAGAS(Es et al., **EACL 2024**)——**免参考** faithfulness = |V|/|S|(答案分解
  为原子陈述集 S,蕴含验证得支持集 V)。

**核心结论**:模型自产引文**必须强制事后核验**,不可信任;忠实度可在句级
(ALCE)与原子级(FActScore/RAGAS)两粒度量化。

**本系统的落地(已实现 + 规划)**:
- 已实现——**双层确定性防线**:`evidence_span` 非原文逐字子串时,抽取层立即
  替换为规则真实句、核验层直接判 `grounded=False`(LLM 无权放行)。这是
  attributable generation 原则**可完全确定性执行**的最低形式。
- 已实现——`eval` 命令对每张卡强制 span 忠实度核验,违例计入回归门。
- 已实现——**RAGAS 式 faithfulness 聚合**:`_rule_verify` 逐术语判断是否见于
  原文(含同义写法),eval 汇总输出 F=|支持要素|/|抽取要素| 作日常回归指标。
- 规划——LLM 后端下可将裁判角色按 **CoVe 四步**实现(初答→生成"此引文出自
  哪部书哪一条?"类验证问题→各自独立走 tcm.py 检索回答→据此修订),验证问答
  与初稿严格隔离。

## 6. 金标准回归评测

**文献**:ALCE 的固定语料+问题+人工可支持性标注的金标准构建法;RAGAS 免参考
持续回归。

**本系统的落地(已实现)**:`eval/gold.jsonl`(用例均先经 `tcm.py` 实检验证后
录入)+ `tcm_graphrag.py eval`,用 provider=rule+tfidf 全确定性路径回归
Recall@k / MRR / 排除正确率 / span 忠实度,失败即 exit 1。**发布门禁**用此金标集,
**日常回归**用 RAGAS 式 faithfulness 聚合,二者互补。

---

## 未验证方向(诚实边界)

调研的五个方向中,**仅方向 1(图谱/混合检索)与方向 2(引用忠实度/幻觉核验)**
有主张通过三票对抗核验。以下三方向的候选主张**未存活**,本系统对应设计
**暂无外部标准背书**,列为公开问题待补充调研:

- **方向 3 — 古汉语/中医 NLP**:SikuBERT / GuwenBERT / EvaHan 系列、TCMBench 等
  中医 LLM 基准的 2024–2026 最新状态未经核验;繁简与异体字归一是否有优于
  `opencc + 内置表`(现 `graphrag/zh.py`)的权威方案,待验证。
  候选源(未核验):aclanthology.org/2025.alp-1.19、arXiv:2406.01126、
  SikuBERT 仓库、BYVoid/OpenCC。
- **方向 4 — 古今术语标准桥接**:ICD-11 第 26 章传统医学模块(TM1 及推进中的
  TM2)、ISO/TC249 术语标准、HPO/SNOMED 与中医证候映射,能否为 **M1–M5 现代
  表型桥接层**提供可直接复用的编码表,待验证。候选源(未核验):SNOMED 2024
  传统医学工作坊、BMC Med Inform Decis Mak(s12911-022-01913-7)。
- **方向 5 — 证据分级方法学**:GRADE 在非临床试验/古籍文献证据分级上的已发表
  适配、循证中医药学(CPG-TCM 等)对古籍证据分级的共识,以及本系统 **A–E 分级**
  如何与之对齐以获方法学背书,待验证。候选源(未核验):PMID 38490078、
  PMC8904085、PMC7285562。

> **重要**:在方向 4/5 得到核验前,本系统的 M1–M5 桥接与 A–E 分级应被视为
> **面向骨质疏松 MVP 的工程约定**,而非声称与 ICD-11 TM / GRADE 对齐。任何
> 对外表述都不得暗示已获这些国际标准背书。

## 其他核验注记

- HippoRAG 2 的全面优势为**作者自报基准**,尚无独立复现;2026 年已有工作
  (NeocorRAG 等)超越之,它已非绝对天花板。
- NLI/LLM-judge 类忠实度指标与人工判断相关性不完美;FActScore <2% 误差仅在
  英文传记域验证;RAGAS 依赖 LLM judge 的分数稳定性受质疑——故本系统离线
  默认路径用**确定性子串/同义命中**核验,不把 LLM 自评当作唯一忠实度来源。
- 检索融合领域 2025–2026 演进快,本结论以 2026-04 前证据为准。

---

*完整对抗核验记录(113 条抽取主张 → 25 条核验 → 23 条确认 / 2 条否决 →
10 条合成)见 `docs/research-notes/graphrag-sota-2026.md`。*
