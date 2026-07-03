# 平台适配说明

单一事实源:`skills/tcm-ancient-texts/`(Agent Skills 开放格式,SKILL.md + 资源文件)。
三个平台通过不同入口挂接同一份内容,规则与脚本零重复。

## Claude Code

- **项目级**(推荐,随仓库走):`.claude/skills/tcm-ancient-texts` 已软链到
  `skills/tcm-ancient-texts`,打开本仓库即自动可用。
- **用户级**(任何目录可用):`./install.sh claude`
  → 安装到 `~/.claude/skills/`。
- 触发:Claude 根据 SKILL.md frontmatter 的 `description` 自动判断何时加载;
  也可显式说"用 tcm-ancient-texts 技能查……"。
- `CLAUDE.md` 提供项目级常驻约束(引用纪律摘要),与技能互补:
  CLAUDE.md 常驻上下文、极简;SKILL.md 按需加载、详尽。

## OpenClaw

OpenClaw 原生支持 Agent Skills 格式(与 Claude Code 同一规范):

- 工作区级:把 `skills/tcm-ancient-texts/` 放入 `<workspace>/skills/`;
- 全局:`./install.sh openclaw` → `~/.openclaw/skills/`。
- OpenClaw 会读取 SKILL.md frontmatter 决定注入;脚本经其 shell 工具执行,
  与 Claude Code 行为一致。

## Codex(OpenAI)

Codex 无技能系统,但会读取仓库根 `AGENTS.md`:

- `AGENTS.md` 内联了核心约束摘要,并指示 agent 先完整阅读
  `SKILL.md` + `rules/00-core.md` + `rules/30-safety.md`;
- 无需安装步骤;全局使用可把 AGENTS.md 要点合并进 `~/.codex/AGENTS.md`。

## 其他框架(LangGraph / 自研 Agent / MCP)

- 任何能执行 shell 的 agent:把 SKILL.md 全文作为 system prompt 附录 +
  开放 `tcm.py` 调用即可。
- 纯 API 场景:`build --jsonl` 导出 `corpus.jsonl` 自行入库;或按
  `docs/DESIGN.md` 的扩展路线将 `tcm.py` 包装为 MCP server
  (search/get/info 三个 tool,一层薄封装)。

## 一致性保障

- 修改规则只改 `skills/tcm-ancient-texts/` 下的文件;
- `CLAUDE.md` / `AGENTS.md` 只保留"摘要 + 指针",不复制细节,避免漂移。
