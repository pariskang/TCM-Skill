# TCM-Skill · Claude Code 项目指引

本仓库是中医古籍(笈成 CC0 语料,1300+ 部)的「规则 + 技能」包。

## 涉及中医古籍问题时

- 使用 `skills/tcm-ancient-texts/` 技能(已注册为项目技能,见 `.claude/skills/`)。
- 任何古籍引文必须来自 `skills/tcm-ancient-texts/scripts/tcm.py` 的检索结果,
  禁止凭记忆引用。检索词用繁体。
- 医学安全红线见 `skills/tcm-ancient-texts/rules/30-safety.md`,不可覆盖。

## 数据

- `data/` 为生成物(语料克隆 + SQLite 索引),不入库。
- 首次使用:`python3 skills/tcm-ancient-texts/scripts/tcm.py fetch && python3 skills/tcm-ancient-texts/scripts/tcm.py build`
- 快速试用:`fetch --sample 20` 再 `build`。

## 古今表型证据 GraphRAG(证据推理层)

- 证据推理用 `skills/tcm-ancient-texts/scripts/tcm_graphrag.py`(复用 tcm.py 索引)。
- 核心引擎零依赖;LLM 后端可选 `rule`(默认离线)/litellm/azure/poe/openai,SDK 懒加载。
- 本体数据在 `scripts/ontology/*.<domain>.json`;新增病种加三个 JSON 即可。
- 完整设计见 `docs/GRAPHRAG.md`。

## 开发约定

- `tcm.py` 只用 Python 标准库,保持零依赖;改动后用
  `search 桂枝湯` / `get "傷寒論(宋本)" --seq 156 --context 2` 冒烟验证。
- 修改解析器需重新 `build` 并对比段落总数,防止静默丢文。
- `graphrag/` 引擎核心也保持零依赖(仅 LLM 后端可选);改动后用
  `tcm_graphrag.py ask "骨痿 腰膝酸软"`(默认 rule)冒烟,确保离线可跑。
