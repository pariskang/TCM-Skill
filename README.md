# TCM-Skill:中医古籍 · 规则 + 技能包

把「中醫笈成」**1300+ 部 CC0 中医古籍**(《黄帝内经》《伤寒论》《金匮要略》
《本草纲目》……)变成任何 LLM 智能体都能**可靠引用、拒绝幻觉**的本地检索能力。

支持平台:**Claude Code · OpenClaw · Codex**(以及任何能执行 shell 的 agent)。

## 三层设计

| 层 | 内容 | 解决什么 |
|---|---|---|
| **规则** `rules/` | 引文必须来自检索、繁体查询、医学安全红线 | 幻觉引文、临床风险 |
| **技能** `SKILL.md` | 渐进披露的工作流 + 检索策略手册 | 上下文经济性 |
| **工具** `scripts/tcm.py` | 零依赖 CLI:SQLite FTS5 全文检索 | 数亿字语料的毫秒级精确定位 |

## 快速开始

```bash
git clone https://github.com/pariskang/TCM-Skill && cd TCM-Skill

# 1. 拉语料 + 建索引(二选一)
python3 skills/tcm-ancient-texts/scripts/tcm.py fetch --sample 20  # 快速试用
python3 skills/tcm-ancient-texts/scripts/tcm.py fetch             # 全量 1300+ 本
python3 skills/tcm-ancient-texts/scripts/tcm.py build

# 2. 验证
python3 skills/tcm-ancient-texts/scripts/tcm.py search 桂枝湯
python3 skills/tcm-ancient-texts/scripts/tcm.py get "傷寒論（宋本）" --seq 156 --context 2

# 3. 接入你的 agent
./install.sh claude     # Claude Code 用户级安装
./install.sh openclaw   # OpenClaw
# Codex:无需安装,自动读取 AGENTS.md
```

Claude Code 打开本仓库时技能已自动可用(`.claude/skills/` 项目级注册)。

## 效果示例

```
$ tcm.py search "太陽之為病"
- 《傷寒論（宋本）》· 卷第二 / 辨太陽病脉證並治上第五 · 段144
  【太陽之為病】，脉浮，頭項强痛而惡寒。
```

引文自带三要素出处(书·篇·段),agent 的每一条引用都可一键复核;
底本异体字(𤍠、𤼵、隂)原样保留,忠于文献。

## 进阶:古今表型证据 GraphRAG

在检索地基之上叠加**证据推理层**——输入现代疾病/证候/症状组合,返回带**证据等级、
古今表型映射、证候-治法-方药链、模型判断理由与事实核验**的证据卡片:

```bash
# 默认离线(rule provider,无需 API key)
python3 skills/tcm-ancient-texts/scripts/tcm_graphrag.py ask "绝经后骨质疏松 肾虚血瘀 骨痛 活动受限"

# 接入真实 LLM 后端(litellm / azure / poe / openai 可切换)
TCM_LLM_PROVIDER=poe POE_API_KEY=xxx \
  python3 skills/tcm-ancient-texts/scripts/tcm_graphrag.py ask "骨痿 腰膝酸软 老年"
```

四路召回(BM25 / 同义扩展 / **PPR 图谱路径** / **语义向量**)+ **RRF 多路名次融合**
+ 确定性加权重排 + **LLM 交叉编码器精排** + 六个 LLM 角色(查询解析 / 抽取 / 归一 /
精排 / 证据裁判 / 幻觉核验)+ **句级共现排除机制**(《素问》"大骨枯槁…期六月死"
危候判 E 级排除,而《痿论》定义性条文不被同段他句排除词误杀)+ **引用忠实度双层
确定性核验**(evidence_span 必须逐字来自原文)。**简体查询完美支持**、语义召回
离线(TF-IDF)或神经嵌入可切换,默认全离线可跑;`eval` 命令跑**金标准回归评测**
(Recall@k / MRR / 排除正确率,CI 可用)。当前病种 MVP:骨质疏松。
完整设计见 [`docs/GRAPHRAG.md`](docs/GRAPHRAG.md)。

## 仓库结构

```
skills/tcm-ancient-texts/   # 技能(Agent Skills 开放格式,单一事实源)
│  ├── SKILL.md             # 入口:铁律 + 工作流
│  ├── rules/               # 00核心纪律 10检索策略 20引用规范 30安全红线
│  ├── references/          # 语料概况 / 数据格式
│  └── scripts/tcm.py       # fetch / build / search / get / info / list / status
├── .claude/skills/          # Claude Code 项目级注册(软链)
├── CLAUDE.md / AGENTS.md    # 平台常驻摘要(细节永远只在 skills/ 一处)
├── adapters/                # 各平台接入说明
├── docs/DESIGN.md           # 深度设计方案(架构决策、扩展路线图)
└── data/                    # 语料 + 索引(生成物,不入库)
```

## 安全定位

本项目是**文献检索考证工具**:不提供诊疗建议、不换算古方剂量、
毒性药物强制警示。详见 `skills/tcm-ancient-texts/rules/30-safety.md`。

## 致谢与授权

- 语料:[中醫笈成](https://jicheng.tw)(公有领域典籍以 CC0 释出,
  [著作权声明](https://jicheng.tw/tcm/copyright.html));本仓库不再分发文本,
  由使用者从上游 [gitlab.com/jicheng/jc.data](https://gitlab.com/jicheng/jc.data) 拉取。
- 本仓库代码与文档:MIT License。
