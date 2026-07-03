---
name: tcm-ancient-texts
description: 检索、引用、考证 1300+ 部中医古籍原文(《黄帝内经》《伤寒论》《金匮要略》《本草纲目》《温病条辨》等,笈成 CC0 语料)。当用户询问中医经典条文原文、方剂组成与出处、药物本草记载、古籍版本考证、跨书条文对比、经络腧穴文献时使用。仅作文献检索考证,不提供临床诊疗建议。
---

# 中医古籍检索技能 (TCM Ancient Texts)

基于「中醫笈成」CC0 开放语料(1300+ 部典籍,数亿字)的本地全文检索系统。
核心工具:`scripts/tcm.py`(纯 Python 标准库,SQLite FTS5 索引)。

## 铁律(违反即失败)

1. **禁止凭记忆引用原文**。所有古籍引文必须来自 `tcm.py search/get` 的实际输出。
   记忆中的条文可能是错的、混淆版本的、或根本不存在的。宁可回答"库中未检得",
   不可编造引文。
2. **每条引文必须带出处**:`《书名》· 篇章 · 段号`(工具输出自带,原样保留)。
3. **查询必须用繁体字**。语料为繁体竖排转横排文本;简体查询会漏检。
   自行将用户的简体词转为繁体再检索(汤→湯、证→證、脉→脈)。
4. **文献 ≠ 医嘱**。涉及方药剂量、毒性药材(附子、烏頭、砒霜、硃砂、雄黃、馬錢子等)
   时,必须声明内容仅为古籍文献记载,不构成用药建议。详见 `rules/30-safety.md`。

## 工作流

### 首次使用:检查环境

```bash
python3 scripts/tcm.py status
```

若索引缺失,按提示构建(全量约需 10 分钟,一次性):

```bash
python3 scripts/tcm.py fetch            # 全量:git 浅克隆 gitlab.com/jicheng/jc.data
python3 scripts/tcm.py fetch --sample 20  # 或:仅抓 20 本快速试用
python3 scripts/tcm.py build            # 解析 + 建 FTS5 索引
```

### 日常检索(标准三步)

```bash
# 1. 检索:定位条文(≥3 字走 FTS5,<3 字自动回退精确子串)
python3 scripts/tcm.py search "太陽之為病" --limit 10
python3 scripts/tcm.py search 石膏 --book 本草綱目

# 2. 取文:按段号取完整段落,--context N 取前后文(方剂要看组成+煎服法,常需 --context 2)
python3 scripts/tcm.py get "傷寒論（宋本）" --seq 156 --context 2
python3 scripts/tcm.py get 溫病條辨 --section 上焦篇 --limit 20

# 3. 核对元数据:引用前确认版本底本
python3 scripts/tcm.py info 溫病條辨
```

### 浏览书目

```bash
python3 scripts/tcm.py list --dynasty 清 --limit 30
python3 scripts/tcm.py list --keyword 傷寒
```

## 检索技巧速查

- **命中为空时**依次尝试:① 确认繁体 ② 缩短查询词(如"桂枝湯主之"→"桂枝湯")
  ③ 换异体写法(藏/臟、痺/痹、欬/咳、鬱/郁) ④ 拆词分别检索再交叉。
- **书名精确匹配**:`get`/`info` 需要库内准确书名。不确定时先
  `list --keyword 傷寒` 查全名(注意《傷寒論（宋本）》带全角括号、
  同名书有 `_1` 后缀区分版本)。
- **跨书对比**:同一条文分别在不同书内 `search --book` 检索,对比用字差异,
  并在回答中注明各自底本(`info` 可查)。
- 更多策略见 `rules/10-retrieval.md`;引用格式规范见 `rules/20-citation.md`。

## 回答产出规范

- 引文用引号或引用块,后附出处;自己的解读与原文严格分开。
- 说明检索范围:"检索了库内 N 处提及"或"仅查《某书》"。
- 库中未收录的书(如现代教材)明确说"不在本语料库内",不要用别的书冒充。
- 涉及临床话题时附一句:"以上为古籍文献记载,仅供文献研究,不构成医疗建议。"

## 进阶:古今表型证据 GraphRAG(证据推理层)

当任务不是"查条文原文"而是**"哪些古籍条文能作为某现代疾病/证候研究的证据"**
(古今表型映射、证据链、证据分级、跨书证据发现)时,用 `scripts/tcm_graphrag.py`:

```bash
# 默认离线(rule provider,无需 API key),输出带等级/映射/核验的证据卡片
python3 scripts/tcm_graphrag.py ask "绝经后骨质疏松 肾虚血瘀 骨痛 活动受限"
python3 scripts/tcm_graphrag.py ask "骨痿 腰膝酸软" --format json   # 供下游消费
python3 scripts/tcm_graphrag.py domains                            # 可用病种本体
```

它复用同一份 SQLite 索引,叠加:古今术语本体、**四路召回**(lexical/synonym/graph/
**语义向量**)、加权重排、**LLM 交叉编码器精排**、六个 LLM 角色(查询解析/抽取/归一/
精排/证据裁判/幻觉核验)、排除机制与证据卡片。**简体查询完美支持**(opencc/内置表/
LLM 三重归一)。可切换 LLM 后端 `--provider rule|litellm|azure|poe|openai`、语义后端
`--semantic tfidf|openai|...` 提升质量;默认 `rule`+`tfidf` 全离线可跑。
**引用铁律同样适用**:证据卡片的原文与出处来自检索,不可脱离工具编造。
详见 `docs/GRAPHRAG.md`(在仓库根)。当前病种 MVP:骨质疏松。

## 参考文档(按需查阅)

- `rules/00-core.md` — 核心行为纪律(完整版)
- `rules/10-retrieval.md` — 检索策略手册(异体字表、常见坑)
- `rules/20-citation.md` — 引用与考证格式规范
- `rules/30-safety.md` — 医学安全红线(毒性药物表、十八反十九畏)
- `references/corpus.md` — 语料概况、书目分类、授权说明
- `references/data-format.md` — 数据格式(HTML schema / SQLite / JSONL)
- `references/classical-20.md` — **二十经典导览**:各书内容摘要、检索策略、主题速查表(经子进程实际检索验证)
- `docs/GRAPHRAG.md`(仓库根)— 证据推理层完整设计与用法
