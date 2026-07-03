# TCM-Skill · Agent 指引(Codex / 通用智能体)

本仓库提供中医古籍检索的「规则 + 技能」包(1300+ 部典籍,CC0 语料)。

**处理任何中医古籍相关任务前,先完整阅读:**

1. `skills/tcm-ancient-texts/SKILL.md` — 工作流与工具用法(必读)
2. `skills/tcm-ancient-texts/rules/00-core.md` — 行为纪律(必读)
3. `skills/tcm-ancient-texts/rules/30-safety.md` — 医学安全红线(必读)

其余 `rules/` 与 `references/` 文档按需查阅。

## 核心约束(摘要)

- 古籍引文只能来自 `python3 skills/tcm-ancient-texts/scripts/tcm.py search|get`
  的实际输出,并保留其出处格式(《书名》·篇章·段号)。禁止凭记忆引用。
- 检索词必须使用繁体中文。
- 文献检索不构成医疗建议;涉及方药需附免责声明。

## 环境准备

```bash
python3 skills/tcm-ancient-texts/scripts/tcm.py status   # 检查
python3 skills/tcm-ancient-texts/scripts/tcm.py fetch    # 拉语料(全量;--sample 20 试用)
python3 skills/tcm-ancient-texts/scripts/tcm.py build    # 建索引
```

依赖:Python 3.9+(标准库即可),SQLite ≥ 3.34(FTS5 trigram)。
