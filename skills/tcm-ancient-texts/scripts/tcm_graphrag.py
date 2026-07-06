#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcm_graphrag.py — 中医古籍表型证据 GraphRAG 系统 CLI。

在 tcm.py 检索地基之上,提供"古今表型证据匹配与可解释判断":输入现代疾病/
表型/证候/症状组合,返回带证据等级、古今表型映射、证候-治法-方药链、模型判断
理由与事实核验的证据卡片。

LLM 后端:rule(离线,默认)| litellm | azure | poe | openai。
详见 docs/GRAPHRAG.md。

子命令:
  ask        证据检索判断,输出证据卡片(markdown/json)
  eval       金标准回归评测(Recall@k/MRR/排除正确率/引用忠实度)
  providers  查看/自检 LLM provider 连通性
  domains    列出可用病种本体
  config     打印当前生效配置

示例:
  python3 tcm_graphrag.py ask "绝经后骨质疏松 肾虚血瘀 骨痛 活动受限"
  TCM_LLM_PROVIDER=poe POE_API_KEY=... python3 tcm_graphrag.py ask "骨痿 腰膝酸软" --provider poe --model GPT-4o
  python3 tcm_graphrag.py providers --check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphrag.config import load_config, ROLES          # noqa: E402
from graphrag import ontology as onto_mod                # noqa: E402


def _overrides(args) -> dict:
    llm = {}
    for k in ("provider", "model", "api_base", "api_version", "deployment"):
        v = getattr(args, k, None)
        if v:
            llm[k] = v
    if getattr(args, "temperature", None) is not None:
        llm["temperature"] = args.temperature
    ov = {}
    if llm:
        ov["llm"] = llm
    if getattr(args, "domain", None):
        ov["domain"] = args.domain
    if getattr(args, "topk", None):
        ov["topk_cards"] = args.topk
    # 语义召回后端
    sem = {}
    if getattr(args, "semantic", None):
        sem["provider"] = args.semantic
    if getattr(args, "embed_model", None):
        sem["provider"] = sem.get("provider") or "openai"
        sem["model"] = args.embed_model
    if sem:
        ov["semantic"] = sem
    # LLM 精排开关
    if getattr(args, "no_llm_rerank", False):
        ov["llm_rerank"] = False
    if getattr(args, "llm_rerank", False):
        ov["llm_rerank"] = True
    return ov


def cmd_ask(args):
    cfg = load_config(args.config_file, _overrides(args))
    from graphrag.pipeline import GraphRAG
    from graphrag.llm import LLMError
    try:
        engine = GraphRAG(cfg, verbose_build=args.verbose)
    except (FileNotFoundError, LLMError) as e:
        sys.exit(f"[error] {e}")
    try:
        result = engine.ask(args.query, book=args.book, verbose=args.verbose)
    finally:
        engine.close()

    cards = result["cards"]
    if args.format == "json":
        from graphrag.evidence import cards_to_json
        print(cards_to_json(result["query_analysis"], cards))
        return

    qa = result["query_analysis"]
    print(f"# 古籍证据检索：{args.query}\n")
    print(f"**解析** — 现代疾病:{'、'.join(qa.get('modern_disease', [])) or '—'}"
          f" | 现代表型:{'、'.join(qa.get('modern_phenotypes', [])) or '—'}"
          f" | 证候:{'、'.join(qa.get('tcm_patterns', [])) or '—'}")
    print(f"**检索词** — {'、'.join(qa.get('ancient_terms', []))}")
    print(f"**LLM 后端** — {cfg.llm.provider}"
          + (f"（{cfg.llm.model or cfg.llm.deployment}）" if cfg.llm.provider != "rule" else "（离线规则引擎）"))
    rr = "开" if (cfg.llm_rerank and cfg.llm.provider != "rule") else "关"
    if cfg.semantic.provider in ("off", "none", ""):
        sem_label = "关"
    elif cfg.semantic.provider == "tfidf":
        sem_label = "tfidf（char-ngram TF-IDF,离线）"
    else:
        sem_label = f"{cfg.semantic.provider}（{cfg.semantic.model or '神经嵌入'}）"
    print(f"**语义召回** — {sem_label} | **LLM 精排** — {rr}")
    print(f"**候选/产出** — 召回重排 {result.get('n_candidates', 0)} 条,"
          f"输出 {len(cards)} 张证据卡片\n")
    if not cards:
        print(result.get("note", "无命中。"))
        return
    for i, c in enumerate(cards, 1):
        print(f"\n---\n\n**[{i}]** {c.to_markdown()}")
    print("\n---\n")
    print("> 以上为古籍文献的检索与证据判断,仅供文献研究,不构成医疗建议。"
          "证据等级由模型/规则初判,专家复核后方可用于科研结论。")


def cmd_providers(args):
    cfg = load_config(args.config_file, _overrides(args))
    print(f"当前 provider: {cfg.llm.provider}")
    print(f"默认模型: {cfg.llm.model or cfg.llm.deployment or '(未设置)'}")
    if cfg.llm.role_models:
        print(f"角色模型覆盖: {cfg.llm.role_models}")
    print("\n可选 provider: rule(离线) / litellm / azure / poe / openai")
    print("环境变量:")
    print("  通用    TCM_LLM_PROVIDER, TCM_LLM_MODEL, TCM_LLM_TEMPERATURE")
    print("  azure   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
          "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT")
    print("  poe     POE_API_KEY (端点默认 https://api.poe.com/v1)")
    print("  openai  OPENAI_API_KEY, OPENAI_BASE_URL")
    print("  litellm TCM_LLM_MODEL(如 gpt-4o / claude-3-5-sonnet / gemini/…), TCM_LLM_API_KEY")
    if args.check:
        from graphrag.llm import probe
        print("\n连通性自检:")
        print("  " + probe(cfg.llm))


def cmd_domains(args):
    doms = onto_mod.available_domains()
    print("可用病种本体:")
    for d in doms:
        o = onto_mod.load_ontology(d)
        print(f"  - {d}: {o.label}（{len(o.terms)} 术语,{len(o.bridges)} 桥接行,"
              f"{len(o.edges)} 图谱边）")


def cmd_eval(args):
    cfg = load_config(args.config_file, _overrides(args))
    from graphrag.evaluate import run_eval
    from graphrag.llm import LLMError
    try:
        ok = run_eval(cfg, gold_path=args.gold, k=args.k)
    except (FileNotFoundError, LLMError) as e:
        sys.exit(f"[error] {e}")
    if not ok:
        sys.exit(1)


def cmd_config(args):
    cfg = load_config(args.config_file, _overrides(args))
    print(json.dumps({
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model,
                "api_base": cfg.llm.api_base, "deployment": cfg.llm.deployment,
                "temperature": cfg.llm.temperature, "max_tokens": cfg.llm.max_tokens,
                "role_models": cfg.llm.role_models},
        "domain": cfg.domain, "topk_recall": cfg.topk_recall,
        "topk_cards": cfg.topk_cards, "weights": cfg.weights or "(默认)",
        "semantic": {"provider": cfg.semantic.provider, "model": cfg.semantic.model,
                     "topk": cfg.semantic.topk},
        "llm_rerank": cfg.llm_rerank, "llm_rerank_topn": cfg.llm_rerank_topn,
    }, ensure_ascii=False, indent=2))


def _add_common(p):
    p.add_argument("--provider", help="rule/litellm/azure/poe/openai")
    p.add_argument("--model")
    p.add_argument("--api-base", dest="api_base")
    p.add_argument("--api-version", dest="api_version")
    p.add_argument("--deployment")
    p.add_argument("--temperature", type=float)
    p.add_argument("--domain")
    p.add_argument("--config-file", dest="config_file")
    p.add_argument("--semantic", choices=["off", "tfidf", "litellm", "openai", "azure"],
                   help="语义召回后端(默认 tfidf 离线;神经嵌入需对应 SDK)")
    p.add_argument("--embed-model", dest="embed_model",
                   help="神经嵌入模型(如 text-embedding-3-small / bge-m3)")
    p.add_argument("--llm-rerank", dest="llm_rerank", action="store_true",
                   help="强制开启 LLM 交叉编码器精排")
    p.add_argument("--no-llm-rerank", dest="no_llm_rerank", action="store_true",
                   help="关闭 LLM 精排")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="证据检索判断")
    p.add_argument("query")
    p.add_argument("--book", help="限定某书")
    p.add_argument("--topk", type=int, help="输出证据卡片数")
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.add_argument("--verbose", "-v", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("providers", help="查看/自检 LLM provider")
    p.add_argument("--check", action="store_true", help="发一次请求测试连通性")
    _add_common(p)
    p.set_defaults(func=cmd_providers)

    p = sub.add_parser("domains", help="列出病种本体")
    _add_common(p)
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("eval", help="金标准回归评测(Recall@k/MRR/排除正确率/引用忠实度)")
    p.add_argument("--gold", help="金标准 JSONL(默认 eval/gold.jsonl)")
    p.add_argument("--k", type=int, default=10, help="top-k(默认 10)")
    _add_common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("config", help="打印生效配置")
    _add_common(p)
    p.set_defaults(func=cmd_config)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
