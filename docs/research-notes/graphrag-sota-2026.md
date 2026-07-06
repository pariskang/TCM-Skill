# GraphRAG × 中医古籍证据系统:2024–2026 SOTA 调研核验记录

> 本文件是 deep-research 对抗核验流程(fan-out 检索 → 抓取 → 三票对抗核验 →
> 合成)的原始产物存档,供 `docs/RESEARCH.md` 的结论回溯。每条 finding 附
> 逐字证据与投票记录;未通过核验的主张列于文末 refuted。

**统计** — 检索角度 5;抓取来源 23;抽取主张 113;核验 25;确认 23;否决 2;合成 10;agent 调用 105

## 执行摘要

针对该中医古籍证据 GraphRAG 系统,2024–2026 年的顶级成果在方向 1(图谱增强与混合检索)和方向 2(引用忠实度与幻觉核验)上产出了 10 项经三票对抗核验的高置信发现:HippoRAG 2(ICML 2025)证明纯图谱召回会损害基础事实检索、必须与稠密段落检索深度融合,LightRAG(Findings of EMNLP 2025)的双层(实体级/主题级)检索与图-向量一体化直接对应本系统四路召回架构;混合检索基准(2026)证实 BM25 词法通道在专业领域仍应保持一等公民地位,RRF(k=60,Cormack 2009)是生产标准融合基线,但调优的加权分数融合在质量上可胜过 RRF——支持保留本系统的确定性加权重排并以 RRF 作消融基线。幻觉核验方面,ALCE 的 NLI 引文召回/精确率、FActScore 的原子事实分解(自动估计器误差<2%)、CoVe 四步自验证(Findings of ACL 2024)与 RAGAS 的免参考忠实度 F=|V|/|S|(EACL 2024)共同构成可直接落地的引用核验与回归评测配方;GPT-4 级模型在开放问答上约 50% 生成未被所引段落完全支持,证明事后蕴含核验必须强制化。方向 3–5(古汉语 NLP 资源、ICD-11 TM/ISO TC249 术语桥接、GRADE 证据分级适配)的候选主张未通过对抗核验存活,本报告无法就此给出已验证结论,需补充调研。

## 已验证发现(通过三票对抗核验)

### 1. 图谱召回不可作唯一通道:HippoRAG 2 之前的结构增强 RAG(含各类 KG-based GraphRAG)虽提升 sense-making 与关联性,但在基础事实记忆任务上显著低于标准向量 RAG;HippoRAG 2(Gutiérrez et al., ICML 2025)通过在 Personalized PageRank 之上加入更深的段落整合与更有效的在线 LLM 使用,才在事实、sense-making、关联(多跳)三类任务上全面超越标准 RAG,并较 SOTA 稠密嵌入模型(NV-Embed-v2)在关联记忆任务上提升 7%。对本系统的落地点:graph 通道应以查询实体为种子跑 PPR、且必须与语义/词法通道加权融合而非独立成路,融合时保留段落级(而非仅三元组级)证据。

- **置信度**:high　**投票**:3-0 / 3-0 / 3-0(三条主张合并)
- **来源**:https://arxiv.org/abs/2502.14802 ; https://arxiv.org/abs/2502.11371 ; https://arxiv.org/abs/2506.05690
- **逐字证据**:原文摘要逐字核验:"their performance on more basic factual memory tasks drops considerably below standard RAG"; "HippoRAG 2 builds upon the Personalized PageRank algorithm ... achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model"。独立评测(arXiv:2502.11371:GraphRAG 在 NQ 单跳事实 QA 上比 vanilla RAG 低约 13.4%)佐证图谱方法仅在多跳/sense-making 上占优。ICML 2025 同行评审,代码开源(OSU-NLP-Group/HippoRAG)。

### 2. LightRAG(Guo, Xia, Yu, Ao, Huang;arXiv:2410.05779,2024-10 首发,Findings of EMNLP 2025)将图结构直接并入文本索引与检索过程,采用双层检索——低层(具体实体/关系)+ 高层(宏观主题),并将图结构与向量表征一体化以高效检索实体及其关系。对本系统的落地点:四路召回中 lexical/synonym 对应低层实体键、graph 对应关系与主题层;可仿照其做法为本体 L1-L7 节点与关系分别建向量键,再经图结构扩展,而非把 graph 与 semantic 当作互不通信的独立通道。

- **置信度**:high　**投票**:3-0 / 3-0 / 3-0(三条主张合并)
- **来源**:https://arxiv.org/abs/2410.05779 ; https://aclanthology.org/2025.findings-emnlp.568
- **逐字证据**:摘要逐字核验:"graph structures into text indexing and retrieval processes"; "a dual-level retrieval system that enhances comprehensive information retrieval from both low-level and high-level knowledge discovery"; "the integration of graph structures with vector representations facilitates efficient retrieval of related entities and their relationships"。正文确认低层=实体及属性/关系、高层=主题与概括,local keywords 匹配实体、global keywords 匹配关系。注意限定:LightRAG 的实体抽取依赖较强 LLM(约 32B+),对零依赖离线路径需用规则/本体替代抽取。

### 3. GraphRAG 综述(Zhang et al., arXiv:2501.13958,2025,2025-09 修订)将该领域系统化为三大创新:(i) 显式刻画实体关系与领域层级的图结构知识表示,(ii) 支持多跳推理、保留上下文的图检索,(iii) 结构感知的知识整合生成。三者分别对应本系统的 L1-L7 本体层、graph 召回路、以及答案生成时携带结构化证据链——可用作系统架构对标与相关工作定位的权威框架。

- **置信度**:high　**投票**:3-0
- **来源**:https://arxiv.org/abs/2501.13958
- **逐字证据**:摘要逐字核验三大创新表述。单一来源但为领域综述、2025-09 仍在更新,三票全票通过;其作用是架构对标框架而非实验性结论,来源强度与主张强度匹配。

### 4. 两阶段流水线(BM25+dense 混合召回 → 神经重排)是当前最佳实践:在 T2-RAGBench 金融文本+表格基准(23,088 查询 / 7,318 文档)上达 Recall@5=0.816、MRR@3=0.605,大幅超越所有单阶段方法;且 BM25 在专业领域单独胜过 SOTA 稠密检索(MRR@3 0.411 vs 0.351,Recall@5 0.644 vs 0.587),其混合采用 RRF、平滑常数 k=60。对本系统的落地点:(1) SQLite FTS5 词法通道保持一等公民,不被嵌入替代——古籍中方名、药名、条文编号等精确术语与金融 ticker 同理利于词法匹配;(2) 在确定性加权重排后加可选 LLM 交叉编码器精排的现有设计与 SOTA 流水线结构一致;(3) 以 RRF(k=60) 作为对照基线做融合消融。

- **置信度**:medium　**投票**:3-0 / 3-0 / 3-0(三条主张合并)
- **来源**:https://arxiv.org/html/2604.01733v1 ; https://arxiv.org/abs/2506.12071
- **逐字证据**:原文逐字核验:"a two-stage pipeline combining hybrid retrieval with neural reranking achieves Recall@5 of 0.816 and MRR@3 of 0.605, outperforming all single-stage methods by a large margin"; "BM25 outperforms state-of-the-art dense retrieval on financial documents"; RRF k=60 与原始论文取值一致。置信度降为 medium 的原因:2026-04 preprint 未经同行评审,且为金融领域单一基准,向文言文语料的迁移属类比而非实证;但『混合优于单路』与 BEIR(Thakur et al. 2021)零样本结论及 2025 年多项工作一致。

### 5. RRF 是免分数归一化的生产标准融合方法:经典公式 score(d)=Σ 1/(k+rank),默认 k=60(源自 Cormack, Clarke & Büttcher, SIGIR 2009;k 越大越平坦、越小越突出头部),已被 OpenSearch 2.19(2025-02 GA)作为 Neural Search 插件内置混合检索融合技术(合并词法/k-NN/神经查询的排名列表),Elasticsearch 同样内置;在 TREC-COVID(171,332 篇 / 50 专家查询)上 SPLADE 稀疏 + BGE 稠密的 RRF 融合取得 nDCG@10=0.828,超稠密单路 6.1%、稀疏单路 14.9%。对本系统的落地点:RRF 天然支持任意路数排名融合,可直接套用于四路召回,作为现有加权融合的零调参对照与回退方案。

- **置信度**:high　**投票**:3-0 / 3-0 / 3-0(三条主张合并)
- **来源**:https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/ ; https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf ; https://arxiv.org/pdf/2604.13728 ; https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/
- **逐字证据**:公式与 k=60 对照 SIGIR 2009 原始论文逐字核验("k = 60 was fixed during a pilot investigation");OpenSearch 2.19.0(2025-02-11)发布说明、官方文档与公开 RFC(neural-search #659/#865)确认为 GA 功能而非实验特性;TREC-COVID 数字经 Table I 复算(0.8282 vs 0.7803/0.7203,+6.14%/+14.98%)。注意:TREC-COVID 结果来自单作者未评审 preprint,仅限该基准;曾有一条『RRF 三方面鲁棒性优于分数归一化』的主张被否决(1-2),不应引用该表述。

### 6. 调优的加权分数融合在质量上可胜过 RRF——支持保留本系统现有的确定性加权重排:OpenSearch 自家六数据集(BEIR 子集)基准显示 RRF 的 NDCG@10 比分数归一化融合低 3.86%,仅换来 p50/p90/p99 延迟 1.62%/1.42%/0.78% 的微小改善;同行评审工作(Bruch et al., ACM TOIS 2023)亦证明归一化分数的凸组合在所有测试语料上(域内与域外)优于 RRF。对本系统的落地点:不要盲目改用 RRF;继续用可调权重的加权融合并以离线金标准集调参,RRF 仅作无调参数据时的稳健基线。

- **置信度**:medium　**投票**:3-0
- **来源**:https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/ ; https://dl.acm.org/doi/10.1145/3596512
- **逐字证据**:厂商博客(2025-02-12)逐字核验四个数字与六数据集范围,且是厂商自曝新功能质量劣势、反向排除营销偏差;Bruch et al. TOIS 2023 提供同行评审佐证。降为 medium 因主来源为博客,且 OpenSearch 对比用的是其默认归一化管线而非手工调优重排(主张中 "can outperform" 的措辞已作对冲)。

### 7. ALCE(Gao, Yen, Yu & Chen, EMNLP 2023)是引文忠实度可复现评测的模板:由 ASQA(长式事实)/QAMPARI(列表)/ELI5(解释)三数据集构成,覆盖 Wikipedia(21M 段落)与 Sphere 网页级(899M 段落)语料;用 NLI 蕴含模型(TRUE,T5-11B)自动计算引文召回(每条陈述须被其所引段落拼接蕴含)与引文精确率(每条引文须单独相关);并揭示即使 ChatGPT/GPT-4,在 ELI5 上约 50% 生成未被所引段落完全支持。对本系统的落地点:(1) 幻觉核验角色按『陈述级引文召回/精确率』二指标实现,以 tcm.py 检索段落为蕴含前提;(2) 金标准回归集仿 ALCE 构建(固定语料 + 问题 + 人工标注可支持性);(3) 模型自产引文一律强制事后蕴含核验,不可信任。

- **置信度**:high　**投票**:3-0 / 3-0 / 3-0(三条主张合并)
- **来源**:https://ar5iv.labs.arxiv.org/html/2305.14627 ; https://aclanthology.org/2023.emnlp-main.741/ ; https://github.com/princeton-nlp/ALCE
- **逐字证据**:三条主张全部对照原文逐字核验:数据集构成、φ(concat(C_i), s_i)=1 的召回定义、TRUE T5-11B、"even the best models lack complete citation support 50% of the time"。2024-2025 后续工作(LongCite arXiv:2409.02897、FineRef AAAI 2025)确认引文忠实度仍是持续失败模式。实现注意:精确率的『无关』判定是两条件测试(单独不蕴含且冗余),不可简化为逐引文蕴含;NLI 自动评测与人工判断相关性不完美。

### 8. FActScore(Min et al., EMNLP 2023)提供原子事实粒度的事实精确率度量:将长文本生成分解为原子事实、计算被可靠知识源支持的比例;其『检索 + 强 LLM』自动估计器复现人工 FActScore 误差 <2%。对本系统的落地点:GraphRAG 答案核验可在 ALCE 句级之下再细化到原子主张粒度,以固定古籍语料为唯一知识源计算支持率;自动估计器模式即本系统『LLM 幻觉核验』角色的已验证模板(领域适配:文言分解 prompt 需自行校准,MedScore 2025/VeriScore 2024 已示范换域可行)。

- **置信度**:high　**投票**:3-0 / 3-0(两条主张合并)
- **来源**:https://aclanthology.org/2023.emnlp-main.741/ ; https://arxiv.org/abs/2406.19276
- **逐字证据**:ACL Anthology 摘要逐字核验:"breaks a generation into a series of atomic facts and computes the percentage of atomic facts supported by a reliable knowledge source"; "an automated model that estimates FACTSCORE using retrieval and a strong language model, with less than a 2% error rate"。注意:仅测精确率不测覆盖率;<2% 误差是在人物传记/Wikipedia 域验证的,古文域未测;后续工作指出分解环节存在过度分解与跨句指代问题(VeriScore、Molecular Facts)。

### 9. Chain-of-Verification(Dhuliawala et al., Meta AI / ETH Zürich, Findings of ACL 2024)四步自验证——起草初答 → 规划验证问题 → 独立回答验证问题(避免被初答偏置)→ 生成最终已验证回答——经实验证明在 Wikidata 列表题、闭卷 MultiSpanQA、长文生成等多任务上降低幻觉。对本系统的落地点:LLM 裁判角色可按 CoVe 结构实现:初答后自动生成『此引文出自哪部书哪一条?』类验证问题,每个问题独立走 tcm.py 检索回答,再据此修订终稿;关键工程点是验证问答与初稿隔离。

- **置信度**:high　**投票**:3-0 / 3-0(两条主张合并)
- **来源**:https://aclanthology.org/2024.findings-acl.212/
- **逐字证据**:ACL Anthology 摘要逐字核验四步定义与实验结论("CoVe decreases hallucinations across a variety of tasks, from list-based questions from Wikidata, closed book MultiSpanQA and longform text generation")。同行评审 2024 正式版(原 arXiv:2309.11495),作者机构核实无误。

### 10. RAGAS(Es, James, Espinosa-Anke & Schockaert, EACL 2024 Demo)提供免参考(无需人工金答案)的 RAG 三维回归评测——faithfulness / answer relevance / context relevance;其 faithfulness 的具体配方:LLM 将答案分解为原子陈述集 S,逐条对检索上下文做蕴含验证得支持集 V,得分 F=|V|/|S|。对本系统的落地点:可作为无金标注情况下的持续回归层,每次改动检索/重排/本体后自动跑 faithfulness 与 context relevance;F=|V|/|S| 即幻觉核验角色的可直接编码公式,与 ALCE/FActScore 的人工金标准集互补(RAGAS 日常回归、ALCE 式金标集做发布门禁)。

- **置信度**:high　**投票**:3-0 / 3-0(两条主张合并)
- **来源**:https://aclanthology.org/2024.eacl-demo.16/ ; https://arxiv.org/abs/2309.15217
- **逐字证据**:EACL 2024 正式版摘要与 arXiv 全文逐字核验免参考定位、三维度与 F=|V|/|S| 公式及抽取/验证 prompt。限定:RAGAS 各指标依赖 LLM judge,2024-2025 多项研究质疑 judge 打分稳定性——免参考是以标注成本换裁判可靠性,离线 rule 后端下需自行实现轻量蕴含替代。

## 被否决的主张(对抗核验未存活,不得引用)

- [0-3] The paper uses the standard RRF formula with constant k = 60, citing Cormack, Clarke & Buettcher (SIGIR 2009), confirming that the classic k=60 parameterization remains the default best practice for score-normalization-free rank fusion in 2026. (来源:https://arxiv.org/pdf/2604.13728)
- [1-2] RRF is claimed to be more robust than score-based normalization fusion in three respects: stability under differing score distributions, resistance to outlier scores, and consistency in favoring documents ranked highly by multiple retrievers. (来源:https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)

## 公开问题(方向 3–5 未验证)

- 在文言文/中医古籍语料上,RRF(k=60)与本系统现行确定性加权融合孰优?需按 ALCE 式方法自建带人工可支持性标注的金标准查询集做消融,目前无任何已发表的古文域混合检索融合对比。
- 方向 3 未验证:SikuBERT/GuwenBERT/EvaHan 系列与 TCMBench 等中医 LLM 基准的最新(2024-2026)版本状态如何?繁简与异体字归一是否已有优于 opencc + 内置表的权威方案可替换 graphrag/zh.py?
- 方向 4 未验证:ICD-11 第 26 章 TM 模块(TM1 及 2025 前后推进的 TM2)与 ISO/TC249 术语标准、以及 HPO/SNOMED 与中医证候映射的已发表工作,能否为 M1-M5 现代表型桥接层提供可直接复用的编码表?
- 方向 5 未验证:GRADE 框架在非临床试验的文献/古籍证据分级上有哪些已发表适配方案(循证中医药学界如 CPG-TCM、古籍证据分级共识),A-E 分级如何与之对齐以获得方法学背书?

## 核验注记 / 局限

(1) 覆盖缺口:五个调研方向中仅方向 1(图谱/混合检索)与方向 2(引用忠实度/幻觉核验)有主张通过三票对抗核验;方向 3(GuwenBERT/SikuBERT/EvaHan/TCMBench 及繁简归一)、方向 4(ICD-11 TM1/TM2、ISO/TC249、HPO/SNOMED 证候映射)、方向 5(GRADE 及古籍证据分级)无存活主张,本报告对这三个方向不提供任何已验证结论,系统的 M1-M5 桥接与 A-E 分级设计暂缺外部标准背书。(2) 域迁移风险:混合检索的关键数字(BM25 胜稠密、两阶段流水线 SOTA)来自金融文本+表格与生物医学英文基准,向文言文/中医语料迁移是合理类比而非实证,须以自建金标集复验。(3) 来源强度不均:arXiv:2604.01733 与 arXiv:2604.13728 为未同行评审 preprint(后者单作者),OpenSearch 数据为厂商博客(有官方文档与 RFC 佐证);HippoRAG 2、LightRAG、ALCE、FActScore、CoVe、RAGAS 为同行评审顶会。(4) 自评偏差:HippoRAG 2 的全面优势为作者自报基准,尚无独立复现;2026 年已有工作(NeocorRAG 等)超越之,它已非绝对天花板。(5) 度量本身的局限:NLI/LLM-judge 类忠实度指标与人工判断相关性不完美,FActScore <2% 误差仅在英文传记域验证;RAGAS 依赖 LLM judge 的分数稳定性受质疑。(6) 两条 RRF 相关主张被否决(0-3 与 1-2),涉及『k=60 仍是 2026 默认最佳实践』的引用归属与『RRF 三方面鲁棒性优于分数融合』的表述,引用时应避开这两种说法。(7) 时效:检索融合领域演进快(2025-2026 密集出新),本结论以 2026-04 前证据为准。

## 全部来源

- [primary] https://arxiv.org/abs/2502.14802 — GraphRAG 与混合检索融合 SOTA
- [primary] https://arxiv.org/abs/2410.05779 — GraphRAG 与混合检索融合 SOTA
- [primary] https://arxiv.org/abs/2501.13958 — GraphRAG 与混合检索融合 SOTA
- [primary] https://arxiv.org/html/2604.01733v1 — GraphRAG 与混合检索融合 SOTA
- [primary] https://arxiv.org/pdf/2604.13728 — GraphRAG 与混合检索融合 SOTA
- [primary] https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/ — GraphRAG 与混合检索融合 SOTA
- [primary] https://ar5iv.labs.arxiv.org/html/2305.14627 — 引用忠实度与幻觉核验评测方法
- [primary] https://aclanthology.org/2023.emnlp-main.741/ — 引用忠实度与幻觉核验评测方法
- [primary] https://aclanthology.org/2024.findings-acl.212/ — 引用忠实度与幻觉核验评测方法
- [primary] https://www.researchgate.net/publication/393020278_RAGAs_Automated_Evaluation_of_Retrieval_Augmented_Generation — 引用忠实度与幻觉核验评测方法
- [primary] https://arxiv.org/pdf/2508.15396 — 引用忠实度与幻觉核验评测方法
- [primary] https://arxiv.org/html/2505.04847v2 — 引用忠实度与幻觉核验评测方法
- [primary] https://aclanthology.org/2025.alp-1.19/ — 古汉语与中医领域 NLP 资源
- [primary] https://arxiv.org/abs/2406.01126 — 古汉语与中医领域 NLP 资源
- [primary] https://arxiv.org/html/2511.07148v1 — 古汉语与中医领域 NLP 资源
- [primary] https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing — 古汉语与中医领域 NLP 资源
- [primary] https://github.com/BYVoid/OpenCC — 古汉语与中医领域 NLP 资源
- [secondary] https://link.springer.com/article/10.1186/s12911-022-01913-7 — 古今医学术语标准桥接
- [primary] https://www.snomed.org/news/blog:-snomed-ct-focuses-on-traditional-medicine-at-expo-2024-pre-conference-workshop — 古今医学术语标准桥接
- [primary] https://www.sciencedirect.com/science/article/pii/S2589377722000283 — 古今医学术语标准桥接
- [primary] https://pubmed.ncbi.nlm.nih.gov/38490078/ — 证据分级方法学与古籍证据适配
- [primary] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8904085/ — 证据分级方法学与古籍证据适配
- [primary] https://pmc.ncbi.nlm.nih.gov/articles/PMC7285562/ — 证据分级方法学与古籍证据适配
