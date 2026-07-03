#!/usr/bin/env bash
# 一键把 tcm-ancient-texts 技能安装到各平台的技能目录。
# 用法:
#   ./install.sh claude     # Claude Code(用户级 ~/.claude/skills/)
#   ./install.sh openclaw   # OpenClaw(~/.openclaw/skills/)
#   ./install.sh all
#   ./install.sh claude --copy   # 默认软链接;--copy 改为复制(跨盘/Windows 用)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$HERE/skills/tcm-ancient-texts"
MODE="${1:-all}"
LINK=1
[[ "${2:-}" == "--copy" ]] && LINK=0

install_to() {
  local dest_dir="$1"
  mkdir -p "$dest_dir"
  local dest="$dest_dir/tcm-ancient-texts"
  rm -rf "$dest"
  if [[ $LINK -eq 1 ]]; then
    ln -s "$SKILL_SRC" "$dest"
    echo "已链接: $dest -> $SKILL_SRC"
  else
    cp -r "$SKILL_SRC" "$dest"
    echo "已复制: $dest"
  fi
}

case "$MODE" in
  claude)   install_to "$HOME/.claude/skills" ;;
  openclaw) install_to "$HOME/.openclaw/skills" ;;
  all)      install_to "$HOME/.claude/skills"; install_to "$HOME/.openclaw/skills" ;;
  *) echo "用法: ./install.sh [claude|openclaw|all] [--copy]"; exit 1 ;;
esac

cat <<'EOF'

下一步(首次使用需构建语料索引,任选其一):
  python3 skills/tcm-ancient-texts/scripts/tcm.py fetch --sample 20 && \
  python3 skills/tcm-ancient-texts/scripts/tcm.py build     # 快速试用(20 本)

  python3 skills/tcm-ancient-texts/scripts/tcm.py fetch && \
  python3 skills/tcm-ancient-texts/scripts/tcm.py build     # 全量(1300+ 本,约 10 分钟)

Codex 用户:无需安装,Codex 会自动读取仓库根的 AGENTS.md。
多项目共享索引:export TCM_DATA_DIR=~/.tcm-data
EOF
